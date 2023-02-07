1. Maximum GPU memory usage

>Before loading

    cuda_memory_allocated : 0
    cuda_max_memory_allocated : 0
    cuda_memory_reserved : 0
    cuda_max_memory_reserved : 0
    nvidia 0MB
    Memory : 1.43GB
    
>After loading

    cuda_memory_allocated : 26MB
    cuda_max_memory_allocated : 26MB
    cuda_memory_reserved : 41MB
    cuda_max_memory_reserved : 41MB
    nvidia 529MB
    Memory : 2.77GB
  
>After predict : 

    cuda_memory_allocated : 0.02GB
    cuda_max_memory_allocated : 8.8G
    cuda_memory_reserved : 0.1GB
    cuda_max_memory_reserved : 9GB
    nvidia 1.16GB
    Memory : 9.8GB

>환경 변수

    BREAST_ANALYZER_BATCH_SIZE : batch_size로, GPU_memory와 연관되었습니다.
    BREAST_ANALYZER_WORKER : MultiProcessing할 떄 사용하는 worker의 숫자입니다.

2. 분석시간

2.1측정방법
    
        현재 알고리즘은 이미지를 그리드 형태로 나눈 뒤 총 이미지 패치 갯수에 따라 시간이 비례하게 증가하도록 설계되어 있습니다.
        mpp 0.25, 기준으로 패치 추출에는 0.02-0.025초를 소요하고, 이미지 장 당 추론에는 약 0.05-0.01초 정도를 소요합니다. (정확하진 않고, naive합니다.)
        둘이 합치면 이미지 한 장당 0.025-0.035초 정도를 소요하게 됩니다.
        CPU Memory는 이미지 한 장 당 4MB 정도를 사용합니다.
    
    해당알고리즘은 mpp에 큰 영향을 받게 됩니다.
    OpenSlide에서 read_region으로 patch 영역을 뽑아올 때, 배율이 2배 차이나면, 이미지를 추출할 때 걸리는 시간은 2**2배만큼 차이가 나게 됩니다.
    즉 배율이 6배 차이라면, 패치 추출에 걸리는 시간은 36배 정도 느려지게 됩니다.
    이는 OpenSlide에서 패치를 추출할 때 걸리는 시간이므로, 줄일 수 있는 방법은 오로지 **Data Loader에서 worker**를 늘리는 방법밖에 없습니다.
    
2.2 최대 분석 시간 : 6656초(worker=8), 2444초(worker=40)

    /mnt/nfs0/jycho_external/Visiopharm/28(JPG).tif를 기준으로 worker에 따라 측정된 시간에 2배를 곱한 값입니다. (worker=8, 55분 28초, worker=40, 12분 22초)
    /mnt/nfs0/jycho_external/Visiopharm/28(JPG).tif의 경우에는 이미지의  mpp는0.253이고,
    슬라이드에서는 별다른 배율을 지원하지 않으므로, 한 번 inference할 때 마다 4000size와 16000size의 이미지 패치들을 추출해야 합니다.
    
    high_res_patch의 경우
    원래 mpp2기준 512x512size에서 patch를 뽑아야 하는데, 별다른 배율을 지원하지 않으므로, 4000x4000 size에서 이미지 패치를 추출해서
    512x512 size의 패치를 뽑을 때보다 64(8^2)배의 시간 차이가 발생합니다.
    low_res_patch의 경우
    원래 mpp8기준 512x512size에서 patch를 뽑아야 하는데, 별다른 배율을 지원하지 않으므로, 16000x160000 size에서 이미지 패치를 추출해서
    512x512 xize의 패치를 뽑을 때보다 1028(32^2)배의 시간 차이가 발생합니다.
    
    이미지 패치를 뽑을 때 장당 0.02초가 걸린다고 가정하면, low_respatch와 high_res_patch를 뽑게되면 약 1100배의 시간차이가 발생하게 됩니다.
    이미지 한 장을 처리하는데 약 22초가 걸리고 0.02*1100(22)
    1100개의 이미지 패치를 8개의 프로세스에서 처리하게 되면 1100/8 = (138)
    한개의 프로세스마다 138x22초를 소요하므로, 약 3100초 +-10%의 시간이 걸리게 됩니다.

    이는 worker를 늘릴 때마다 연산 시간이 줄어든다는 의미로, worker를 40까지 늘리게 되면, 이론상으로는 5배 줄어든 600~700초 선으로 연산 시간이 줄어들어야 하지만,
    병목이랑 다른 함수들은 시간 측정이 안 되었으므로 실질적으로는 약 1200초 정도가 걸리게 됩니다.
     
    실제로 돌아가는 환경이 어떻게 되는지는 잘 모르지만, 최대한 worker를 늘리는게 연산시간을 줄일 수 있는 방법입니다.
    (gpu보다는 cpu에서 돌아가는 DataLoader의 병목이 큰 편입니다.)
    
3. Maximum CPU memory usage
>모델 1.4G, WSI는 패치 2100장 기준으로 8.4G정도 사용합니다.

4.지원하는 환경변수들과, 각 값들이 가지는 의미
>"BREAST_ANALYZER_BATCH_SIZE" = 8 -> batch_size를 8로 둡니다. gpu에서 한번에 8개씩 모아서 계산하라는 의미로,
>"BREAST_ANALYZER_WORKER" = 8 -> dataloader에서 Multiprocessing(8 workers)
