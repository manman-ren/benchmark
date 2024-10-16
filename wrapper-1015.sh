# make sure TRITON_KERNEL_DUMP is 1 PEEL_LAST_ITER (PEEL_LAST_ITER only applies when there is a loop_schedule)
# 2 inputs: accuracy override
export ENABLE_PINGPONG=0
git reset --hard HEAD
./check-1015.sh $1 $2 &> 1015.res
export ENABLE_PINGPONG=1
git reset --hard HEAD
./check-1015.sh $1 $2 &> 1015.res2

# change regalloc
export ENABLE_PINGPONG=0
git reset --hard HEAD
patch -p1 < ws-1015/regalloc.patch
./check-1015.sh $1 $2 &> 1015.res3
export ENABLE_PINGPONG=1
git reset --hard HEAD
patch -p1 < ws-1015/regalloc.patch
./check-1015.sh $1 $2 &> 1015.res4
