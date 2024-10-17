import os
import subprocess
import json

# 실험 요약 정보를 확인
def check_experiment_details(model_name, experiment_name):
    details_file = '/app/paper-implementations/vision/experiments/details.json'
    
    # JSON 파일 읽기
    with open(details_file, 'r') as f:
        try:
            details = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error: {details_file} contains invalid JSON.")
    
    # model_name이 존재하는지 확인
    if model_name not in details:
        raise ValueError(f"Error: Model '{model_name}' not found in {details_file}.")
    
    # experiment_name이 해당 model_name에 존재하는지 확인
    if experiment_name not in details[model_name]:
        raise ValueError(f"Error: Experiment '{experiment_name}' not found under model '{model_name}' in {details_file}.")
    
    # summary 출력 (선택 사항)
    # summary = details[model_name][experiment_name].get('summary', 'No summary available.')
    # print(f"Experiment summary for {model_name} - {experiment_name}: {summary}")


# 현재 브랜치가 model_name 브랜치인지 확인
def check_branch(model_name):
    try:
        prev_dir = os.getcwd()
        branch_name = f'experiments/{model_name}'
        os.chdir('/app/paper-implementations/')  # 디렉토리 변경

        # 현재 브랜치 가져오기
        current_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')

        os.chdir(prev_dir)  # 디렉토리 변경
        if current_branch != branch_name:
            raise ValueError(f"Error: Current branch is {current_branch}, not {branch_name}.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during branch check: {e}")

# Git 커밋 및 푸시 수행
def git_push(model_name, experiment_name):
    try:
        prev_dir = os.getcwd()
        branch_name = f'experiments/{model_name}'
        os.chdir('/app/paper-implementations/')  # 디렉토리 변경

        # 파일 추가 및 커밋
        subprocess.check_call(["git", "add", "."])
        commit_message = f"{model_name}-{experiment_name}"
        subprocess.check_call(["git", "commit", "-m", commit_message])

        # 코드 푸시
        subprocess.check_call(["git", "push", "origin", branch_name])

        print("Successfully pushed to git repository.")
        os.chdir(prev_dir)  # 디렉토리 변경
    except subprocess.CalledProcessError as e:
        print(f"Error during git push: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 전체 git push 과정 호출
def git_push_process(model_name, experiment_name):
    # 실험 요약 정보 확인
    check_experiment_details(model_name, experiment_name)

    # 브랜치 확인
    check_branch(model_name)

    # Git 커밋 및 푸시
    git_push(model_name, experiment_name)