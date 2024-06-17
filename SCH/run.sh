ds=(16 32 64 128)

for i in {1..4}
do
    python train.py --Bit ${ds[i-1]} --GID 0 --DS 0
done