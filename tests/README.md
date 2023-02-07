# DeepDx-Analyzer Testing
Analyzer moudle은 크게 세가지 방향으로 테스트를 진행하려고 함.

# 1. input 적절성 확인
input에 적절하지 않은 값을 넣었을때 reject함을 확인 함. 
아래와 같은 내용을 포함함
    - interface level testing (ex. 파일 경로를 input으로 주지 않고 분석 method call -> raise Error)
    - reject unsupported WSI file (ex. x50배 슬라이드 input -> raise Error)
    

# 2. 과거 model 결과와 같은 결과를 내는지 확인
    - 용도 : pre-/post-processing 코드를 수정했을때 이 수정이 output 결과에 영향을 미치지 않는것을 confirm 함
    - 개발팀에서 inference 성능개선을 위해 코드를 수정할 때 주로 사용 됨
    - Expected output은 `tests/snapshots`에, 마지막 test run 의 결과는 `tests/__test_run__`에 저장됨. `tests/__test_run__` 폴더에 있는 json 파일들은 테스트용 WSI파일들에 대한 실행결과로, `spec_2_invariance_test`가 실패할 떄 디버깅용으로 쓸 수 있음  

# 3. ProstateAnalyzerResult 의 각 method에 대한 unittest
    - 용도: 과거 모델과 같은 결과를 내는지 확인하는 E2E test(#2)에서 사용하는 여러 method에 대한 unit test가 필요하며 각 method의 코드를 수정했을 때 이 수정이 적절한 지를 확인 함. 
    - Expected output은 `tests/data`에, 마지막 test run의 결과는 `test/__unittest_run__`에 저장됨. 
    `tests/__unittest_run__` 폴더에 있는 파일들은 테스트용 WSI파일들에 대한 실행결과로, `spec_2_invariance_test`가 실패할 떄 디버깅용으로 쓸 수 있음

# 4. model 경항 산출 (Not implemented)
    - 분석 알고리즘이 변경되었을때 (model을 변경, post-processing변경 등) 변경된 알고리즘의 경향성을 산출하고 visualize 함


* Note: **'1. input 적절성 확인'** 과 **'2. 과거 model 결과와 같은 결과를 내는지 확인'** 중 어디로 넣을지 애매한 test case는, slide metadata만 읽고 판단할 수 있다면 **'1. input 적절성 확인'** 로, 이미지를 읽어야 한다면 **'2. 과거 model 결과와 같은 결과를 내는지 확인'** 으로 분류함
    -  ex. tissue가 없는 WSI를 input으로 넣은 case: metadata로만으로는 판단 할 수 없어 **2. 과거 model 결과와 같은 결과를 내는지 확인** 으로 분류함


## Test Data
TBD