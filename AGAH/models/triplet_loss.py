import paddle
from paddle import nn
from paddle.nn import functional as F

def _euclidean_distances(source, target, squared=False):
    distances = paddle.sum(paddle.pow((source.unsqueeze(1) - target.unsqueeze(0)), 2), axis=-1)
    distances = paddle.clip(distances, min=0)

    if not squared:
        mask = paddle.equal(distances, 0.0).astype('float32')
        distances = distances + mask * 1e-16
        distances = paddle.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances

def _cos_distance(source, target):
    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, axis=-1)
    distances = paddle.clip(1 - cos_sim, min=0)
    return distances

def _get_anchor_triplet_mask(s_labels, t_labels):
    sim = (paddle.matmul(s_labels, paddle.transpose(t_labels, [1, 0])) > 0).astype('float32')
    return sim, 1 - sim

def _get_triplet_mask(s_labels, t_labels):
    sim = (paddle.matmul(s_labels, paddle.transpose(t_labels, [1, 0])) > 0).astype('float32')
    i_equal_j = sim.unsqueeze(2)
    i_equal_k = sim.unsqueeze(1)
    mask = i_equal_j * (1 - i_equal_k)
    return mask

class TripletAllLoss(nn.Layer):
    def __init__(self, dis_metric='euclidean', squared=False, reduction='mean'):
        """
        Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        :param margin:
        :param dis_metric: 'euclidean' or 'dp'(dot product)
        :param squared:
        :param reduction: 'mean' or 'sum' or 'none'
        """
        super(TripletAllLoss, self).__init__()

        self.dis_metric = dis_metric
        self.reduction = reduction
        self.squared = squared

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        if self.dis_metric == 'euclidean':
            pairwise_dist = _euclidean_distances(source, target, self.squared)
        elif self.dis_metric == 'cos':
            pairwise_dist = _cos_distance(source, target)

        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        if self.dis_metric == 'euclidean':
            triplet_loss = anchor_positive_dist - (1 - margin) * anchor_negative_dist
        else:
            triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        # triplet_loss = anchor_positive_dist - anchor_negative_dist

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = _get_triplet_mask(s_labels, t_labels)
        triplet_loss = mask * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = (triplet_loss > 1e-16).astype('float32')
        num_positive_triplets = valid_triplets.sum()

        if self.reduction == 'mean':
            # Get final mean triplet loss over the positive valid triplets
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        elif self.reduction == 'sum':
            triplet_loss = triplet_loss.sum()

        return triplet_loss

class TripletHardLoss(nn.Layer):
    def __init__(self, dis_metric='euclidean', squared=False, reduction='mean'):
        """
        Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        :param margin:
        :param dis_metric: 'euclidean' or 'dp'(dot product)
        :param squared:
        :param reduction: 'mean' or 'sum' or 'none'
        """
        super(TripletHardLoss, self).__init__()

        self.dis_metric = dis_metric
        self.reduction = reduction
        self.squared = squared

    def forward(self, source, s_labels, target=None, t_labels=None, margin=0):
        if target is None:
            target = source
        if t_labels is None:
            t_labels = s_labels

        # Get the pairwise distance matrix
        if self.dis_metric == 'euclidean':
            pairwise_dist = _euclidean_distances(source, target, squared=self.squared)
        elif self.dis_metric == 'cos':
            pairwise_dist = _cos_distance(source, target)

        # First, we need to get a mask for every valid positive (they should have same label)
        # and every valid negative (they should have different labels)
        mask_anchor_positive, mask_anchor_negative = _get_anchor_triplet_mask(s_labels, t_labels)

        # For each anchor, get the hardest positive
        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist = paddle.max(anchor_positive_dist, axis=1, keepdim=True)

        # For each anchor, get the hardest negative
        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = paddle.max(pairwise_dist, axis=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = paddle.min(anchor_negative_dist, axis=1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = paddle.clip(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)

        # Get final mean triplet loss
        if self.reduction == 'mean':
            triplet_loss = paddle.mean(triplet_loss)
        elif self.reduction == 'sum':
            triplet_loss = paddle.sum(triplet_loss)

        return triplet_loss
