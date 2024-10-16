# make sure TRITON_KERNEL_DUMP is 1 PEEL_LAST_ITER
#git reset --hard HEAD
input=$1 # accuracy?
override=$2
echo "Config --------------------- nocst_noCompPipeline_noTMA"
echo "-- no override"
if [ "$input" == "accuracy" ]; then
  ~/projects/triton/clean.sh
  TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
fi
~/projects/triton/clean.sh
rm -rf ~/.triton/dump/*
CUDA_VISIBLE_DEVICES=5 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2 --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
mkdir -p ws-nocst-nocomp-notma
find $TRITON_CACHE_DIR -name "_attn_fwd.ttgir" &> t.res
awk -F "_attn_fwd.ttgir" '{print $1}' t.res &> cache.dir
string1=$(cat cache.dir)
cp $string1/* ws-nocst-nocomp-notma/
#diff ws-nocst-nocomp-notma-override/override-fbcode-no-dot-wait.ttgir ws-nocst-nocomp-notma/_attn_fwd.ttgir

if [ "$override" == "override" ]; then
  find ~/.triton/dump -name "*ttgir" &> t.res
  awk -F "dump" '{print $2}' t.res | awk -F '/' '{print $2}' &> dump.hash
  string1=$(cat dump.hash)
  echo "-- make sure override happens (Overriding kernel with file $string1 ...)"
  mkdir -p ~/.triton/override/$string1/
  # override-no-dot-wait-no-sync.ttgir, override-no-dot-wait-no-sync-no-offset.ttgir
  cp override-fbcode-no-dot-wait.ttgir ~/.triton/override/$string1/_attn_fwd.ttgir
  if [ "$input" == "accuracy" ]; then
    ~/projects/triton/clean.sh
    TRITON_KERNEL_OVERRIDE=1 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
  fi
  ~/projects/triton/clean.sh
  TRITON_KERNEL_OVERRIDE=1  CUDA_VISIBLE_DEVICES=5 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2 --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
  mkdir -p ws-nocst-nocomp-notma-override
  find $TRITON_CACHE_DIR -name "_attn_fwd.ttgir" &> t.res
  awk -F "_attn_fwd.ttgir" '{print $1}' t.res &> cache.dir
  string1=$(cat cache.dir)
  cp $string1/* ws-nocst-nocomp-notma-override/
fi

echo "Config --------------------- nocst_noCompPipeline_TMA"
if [ "$input" == "accuracy" ]; then
  ~/projects/triton/clean.sh
  # accuracy issue due to TMA store for 2nd consumer warp group
  TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
fi
~/projects/triton/clean.sh
rm -rf ~/.triton/dump/*
CUDA_VISIBLE_DEVICES=5 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
mkdir -p ws-nocst-nocomp-tma
find $TRITON_CACHE_DIR -name "_attn_fwd_tma.ttgir" &> t.res
awk -F "_attn_fwd_tma.ttgir" '{print $1}' t.res &> cache.dir
string1=$(cat cache.dir)
cp $string1/* ws-nocst-nocomp-tma/

if [ "$override" == "override" ]; then
  find ~/.triton/dump -name "*ttgir" &> t.res
  awk -F "dump" '{print $2}' t.res | awk -F '/' '{print $2}' &> dump.hash
  string1=$(cat dump.hash)
  echo "-- make sure override happens (Overriding kernel with file $string1 ...)"
  mkdir -p ~/.triton/override/$string1/
  cp ./ws-nocst-nocomp-tma/override.ptx ~/.triton/override/$string1/_attn_fwd_tma.ptx
  if [ "$input" == "accuracy" ]; then
    ~/projects/triton/clean.sh
    TRITON_KERNEL_OVERRIDE=1 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
  fi
  ~/projects/triton/clean.sh
  TRITON_KERNEL_OVERRIDE=1 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
  mkdir -p ws-nocst-nocomp-tma-override
  find $TRITON_CACHE_DIR -name "_attn_fwd_tma.ttgir" &> t.res
  awk -F "_attn_fwd_tma.ttgir" '{print $1}' t.res &> cache.dir
  string1=$(cat cache.dir)
  cp $string1/* ws-nocst-nocomp-tma-override/
fi

############## enable comp pipeline
echo "Config --------------------- nocst_compPipeline_noTMA"
patch -p1 < ws-1015/enable-comp-pipe.patch
if [ "$input" == "accuracy" ]; then
  ~/projects/triton/clean.sh
  TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
fi
~/projects/triton/clean.sh
CUDA_VISIBLE_DEVICES=5 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2 --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
mkdir -p ws-nocst-comp-notma
find $TRITON_CACHE_DIR -name "_attn_fwd.ttgir" &> t.res
awk -F "_attn_fwd.ttgir" '{print $1}' t.res &> cache.dir
string1=$(cat cache.dir)
cp $string1/* ws-nocst-comp-notma/

echo "Config --------------------- nocst_compPipeline_TMA"
# compilation failure if enabling comp pipe without loop_schedule
if [ "$input" == "accuracy" ]; then
  ~/projects/triton/clean.sh
  TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
fi
~/projects/triton/clean.sh
rm -rf ~/.triton/dump/*
CUDA_VISIBLE_DEVICES=5 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
mkdir -p ws-nocst-comp-tma
find $TRITON_CACHE_DIR -name "_attn_fwd_tma.ttgir" &> t.res
awk -F "_attn_fwd_tma.ttgir" '{print $1}' t.res &> cache.dir
string1=$(cat cache.dir)
cp $string1/* ws-nocst-comp-tma/

if [ "$override" == "override" ]; then
  find ~/.triton/dump -name "*ttgir" &> t.res
  awk -F "dump" '{print $2}' t.res | awk -F '/' '{print $2}' &> dump.hash
  string1=$(cat dump.hash)
  echo "-- make sure override happens (Overriding kernel with file $string1 ...)"
  mkdir -p ~/.triton/override/$string1/
  #cp ./ws-tma3-comp/override.ttgir ~/.triton/override/$string1/_attn_fwd_tma.ttgir
  rm ~/.triton/override/$string1/_attn_fwd_tma.ttgir
  cp ws-nocst-comp-tma/override.ptx ~/.triton/override/$string1/_attn_fwd_tma.ptx
  if [ "$input" == "accuracy" ]; then
    ~/projects/triton/clean.sh
    TRITON_KERNEL_OVERRIDE=1 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
  fi
  ~/projects/triton/clean.sh
  TRITON_KERNEL_OVERRIDE=1 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2_tma --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
  mkdir -p ws-nocst-comp-tma-override
  find $TRITON_CACHE_DIR -name "_attn_fwd_tma.ttgir" &> t.res
  awk -F "_attn_fwd_tma.ttgir" '{print $1}' t.res &> cache.dir
  string1=$(cat cache.dir)
  cp $string1/* ws-nocst-comp-tma-override/
fi

echo "Config --------------------- cst_compPipeline_noTMA"
patch -p1 < ws-1015/const.patch
if [ "$input" == "accuracy" ]; then
  ~/projects/triton/clean.sh
  TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
fi
~/projects/triton/clean.sh
rm -rf ~/.triton/dump/*
CUDA_VISIBLE_DEVICES=5 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2 --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
mkdir -p ws-cst-comp-notma
find $TRITON_CACHE_DIR -name "_attn_fwd.ttgir" &> t.res
awk -F "_attn_fwd.ttgir" '{print $1}' t.res &> cache.dir
string1=$(cat cache.dir)
cp $string1/* ws-cst-comp-notma/

if [ "$override" == "override" ]; then
  find ~/.triton/dump -name "*ttgir" &> t.res
  awk -F "dump" '{print $2}' t.res | awk -F '/' '{print $2}' &> dump.hash
  string1=$(cat dump.hash)
  echo "-- make sure override happens (Overriding kernel with file $string1 ...)"
  mkdir -p ~/.triton/override/$string1/
  cp ./ws-cst-swp/override5.ttgir ~/.triton/override/$string1/_attn_fwd.ttgir
  if [ "$input" == "accuracy" ]; then
    ~/projects/triton/clean.sh
    TRITON_KERNEL_OVERRIDE=1 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2,triton_tutorial_flash_v2_notma --num-inputs 1 --seq-len 13 --metrics accuracy --batch 8 --n-heads 16 --d-head 128 --baseline triton_tutorial_flash_v2_notma
  fi
  ~/projects/triton/clean.sh
  TRITON_KERNEL_OVERRIDE=1 CUDA_VISIBLE_DEVICES=5 TORCH_CUDA_ARCH_LIST=9.0a python run_benchmark.py triton --op flash_attention --only triton_tutorial_flash_v2 --num-inputs 1 --seq-len 13 --metrics tflops --batch 8 --n-heads 16 --d-head 128
  mkdir -p ws-cst-comp-notma-override
  find $TRITON_CACHE_DIR -name "_attn_fwd.ttgir" &> t.res
  awk -F "_attn_fwd.ttgir" '{print $1}' t.res &> cache.dir
  string1=$(cat cache.dir)
  cp $string1/* ws-cst-comp-notma-override/
fi
