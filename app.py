# 스트림릿 라이브러리를 사용하기 위한 임포트

import streamlit as st
import pandas as pd
import random


# 세션 상태 초기화 함수
def initialize_session_state() -> None:
    """앱이 처음 실행될 때 필요한 세션 상태를 초기화합니다."""
    if 'word_list' not in st.session_state:
        st.session_state.word_list = load_default_dataset()

    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''

    if 'last_result' not in st.session_state:
        st.session_state.last_result = None  # 'correct' | 'wrong' | None
    if 'order_indices' not in st.session_state:
        st.session_state.order_indices = list(range(len(st.session_state.word_list)))
        random.shuffle(st.session_state.order_indices)
    if 'shuffled' not in st.session_state:
        st.session_state.shuffled = True
    if 'exam_mode' not in st.session_state:
        # exam_mode: 'all' | 'wrong_only'
        st.session_state.exam_mode = 'all'
    if 'current_order_pos' not in st.session_state:
        st.session_state.current_order_pos = 0
    if 'current_real_index' not in st.session_state:
        order = st.session_state.order_indices or list(range(len(st.session_state.word_list)))
        st.session_state.current_real_index = order[st.session_state.current_order_pos % len(order)] if order else 0
    if 'last_question_index' not in st.session_state:
        st.session_state.last_question_index = None
    if 'last_question_word' not in st.session_state:
        st.session_state.last_question_word = None
    if 'last_question_meaning' not in st.session_state:
        st.session_state.last_question_meaning = None


# 기본 데이터셋 로더
def load_default_dataset() -> list[dict]:
    """기본 단어 목록을 반환합니다. 각 항목은 통계 필드를 포함합니다."""
    base_words = [
        {'word': 'apple', 'meaning': '사과'},
        {'word': 'book', 'meaning': '책'},
        {'word': 'computer', 'meaning': '컴퓨터'},
        {'word': 'school', 'meaning': '학교'},
        {'word': 'happy', 'meaning': '행복한'},
    ]

    enriched = []
    for item in base_words:
        enriched.append({
            'word': item['word'],
            'meaning': item['meaning'],
            'correct_count': 0,
            'wrong_count': 0,
            'manual_correct_count': 0,
            'attempt_history': [],  # {'input': str, 'result': 'correct'|'wrong'|'manual'}
        })
    return enriched


# CSV 데이터셋 로더
def load_csv_dataset(uploaded_file) -> list[dict]:
    """업로드된 CSV 파일을 읽어 단어 목록으로 변환합니다.

    지원 형식:
    - 형식 A(일반적): 열 3개(날짜, 영어단어, 뜻), 여러 행, 헤더 유/무
    - 형식 B(요청 형태): 행 3개(1행 날짜들, 2행 영어단어들, 3행 뜻들), 여러 열
    """
    # 우선 헤더 없이 읽기 시도
    try:
        df = pd.read_csv(uploaded_file, header=None)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    # 공백 제거
    df = df.replace({pd.NA: ''}).fillna('')

    # 형식 B(요청): 행이 3개(1행 날짜들, 2행 단어들, 3행 뜻들)인 경우 전치하여 처리
    if df.shape[0] == 3 and df.shape[1] >= 1:
        df = df.T
        df.columns = ['date', 'word', 'meaning']

    # 형식 A: 열이 3개인 경우
    if df.shape[1] == 3:
        # 헤더 추정: 첫 행이 헤더일 수 있음
        possible_headers = {'date', '날짜', 'word', '영어', '단어', 'meaning', '뜻'}
        first_row_values = set(str(v).strip().lower() for v in df.iloc[0].tolist())
        has_header = len(first_row_values & possible_headers) > 0
        if has_header:
            # 헤더 적용을 위해 다시 읽기
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            if df.shape[1] != 3:
                raise ValueError('CSV 형식을 해석할 수 없습니다.')
        else:
            df.columns = ['date', 'word', 'meaning']
    # 형식 변형: 열이 3개가 아니라도, 헤더 이름이 존재할 수 있음
    elif df.shape[1] > 3:
        # 헤더가 있을 것으로 가정하고 주요 컬럼만 추출 시도
        uploaded_file.seek(0)
        df2 = pd.read_csv(uploaded_file)
        lower_cols = [str(c).strip().lower() for c in df2.columns]
        try:
            date_col = lower_cols.index('date') if 'date' in lower_cols else lower_cols.index('날짜')
            word_col = lower_cols.index('word') if 'word' in lower_cols else lower_cols.index('영어') if '영어' in lower_cols else lower_cols.index('단어')
            meaning_col = lower_cols.index('meaning') if 'meaning' in lower_cols else lower_cols.index('뜻')
            df = df2.iloc[:, [date_col, word_col, meaning_col]]
            df.columns = ['date', 'word', 'meaning']
        except Exception:
            raise ValueError('CSV에서 (date/날짜, word/영어/단어, meaning/뜻) 컬럼을 찾을 수 없습니다.')
    else:
        # 열이 1~2개 같은 비정형
        raise ValueError('CSV 형식을 해석할 수 없습니다. 열 3개 또는 3행 구조를 사용하세요.')

    # 레코드 생성
    items: list[dict] = []
    for _, row in df.iterrows():
        date_value = str(row[0]).strip()
        word_value = str(row[1]).strip()
        meaning_value = str(row[2]).strip()
        if not word_value and not meaning_value:
            continue
        items.append({
            'first_learned_at': date_value,
            'word': word_value,
            'meaning': meaning_value,
            'correct_count': 0,
            'wrong_count': 0,
            'manual_correct_count': 0,
            'attempt_history': [],
        })

    if not items:
        raise ValueError('CSV에서 유효한 단어 항목을 찾지 못했습니다.')
    return items


# 정답 판정 유틸리티
def is_answer_correct(user_text: str, answer_text: str) -> bool:
    """사용자 입력과 정답을 소문자/공백 기준으로 비교합니다."""
    normalized_user = (user_text or '').strip().lower()
    normalized_answer = (answer_text or '').strip().lower()
    return normalized_user == normalized_answer


# 시도 결과 반영 함수
def record_attempt(word_item: dict, user_text: str, result: str) -> None:
    """사용자 시도 결과를 단어 항목 통계에 반영합니다."""
    if result == 'correct':
        word_item['correct_count'] += 1
    elif result == 'wrong':
        word_item['wrong_count'] += 1
    elif result == 'manual':
        word_item['manual_correct_count'] += 1
    word_item['attempt_history'].append({'input': user_text, 'result': result})


# 퀴즈 UI 렌더링
def render_quiz_ui() -> None:
    """현재 단어 문제, 입력창, 제출/임의정답/다음 버튼을 렌더링합니다."""
    word_list = st.session_state.word_list
    if not word_list:
        st.info('단어 목록이 비어 있습니다.')
        return

    order = st.session_state.order_indices or list(range(len(word_list)))
    if len(order) == 0:
        st.info('현재 모드에 해당하는 문제가 없습니다. 사이드바에서 모드를 변경하거나 셔플하세요.')
        return
    # 현재 표시 중인 단어 인덱스를 세션에서 고정적으로 사용
    st.session_state.current_real_index = order[st.session_state.current_order_pos % len(order)]
    # 표시할 문제: 채점 전에는 현재 인덱스, 채점 후에는 스냅샷
    if st.session_state.last_result is None:
        display_index = st.session_state.current_real_index
        display_word = word_list[display_index]['word']
        display_meaning = word_list[display_index]['meaning']
    else:
        # 스냅샷이 없다면 안전하게 현재 인덱스를 사용
        display_index = st.session_state.last_question_index if st.session_state.last_question_index is not None else st.session_state.current_real_index
        display_word = st.session_state.last_question_word if st.session_state.last_question_word is not None else word_list[display_index]['word']
        display_meaning = st.session_state.last_question_meaning if st.session_state.last_question_meaning is not None else word_list[display_index]['meaning']

    st.subheader('단어 문제')
    st.markdown(f"- 단어: **{display_word}**")

    # 폼으로 감싸 Enter 키 제출 지원 (채점 전)
    if st.session_state.last_result is None:
        with st.form('quiz_form', clear_on_submit=False):
            st.text_input(
                label='뜻을 입력하세요',
                key='user_input'
            )
            submitted = st.form_submit_button('제출', use_container_width=True)
            if submitted:
                # 채점은 현재 표시 인덱스 기준으로 고정하여 진행
                current_item = word_list[display_index]
                is_correct = is_answer_correct(st.session_state.user_input, current_item['meaning'])
                record_attempt(current_item, st.session_state.user_input, 'correct' if is_correct else 'wrong')
                st.session_state.last_result = 'correct' if is_correct else 'wrong'
                # 스냅샷 저장
                st.session_state.last_question_index = display_index
                st.session_state.last_question_word = current_item['word']
                st.session_state.last_question_meaning = current_item['meaning']
    else:
        # 채점 후: 임의 정답 처리 / 다음 단어 버튼 표시
        col_manual, col_next, _ = st.columns(3)
        with col_manual:
            if st.button('임의 정답 처리', use_container_width=True):
                snapshot_index = st.session_state.last_question_index if st.session_state.last_question_index is not None else display_index
                record_attempt(word_list[snapshot_index], st.session_state.user_input, 'manual')
        with col_next:
            if st.button('다음 단어', use_container_width=True):
                move_to_next()
                st.experimental_rerun()

    if st.session_state.last_result is not None:
        if st.session_state.last_result == 'correct':
            st.success('정답입니다!')
        else:
            st.error(f"오답입니다. 정답: {display_meaning}")
        # 피드백 아래에서도 조작 가능: 임의 정답 처리 / 다음 문제로
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button('임의 정답 처리 (피드백 아래)', key='manual_after_feedback'):
                snapshot_index = st.session_state.last_question_index if st.session_state.last_question_index is not None else display_index
                record_attempt(word_list[snapshot_index], st.session_state.user_input, 'manual')
        with fb_col2:
            if st.button('다음 문제로 (피드백 아래)', key='next_after_feedback'):
                move_to_next()
                st.experimental_rerun()


# 다음 문제로 이동
def move_to_next() -> None:
    """현재 인덱스를 다음으로 이동하고 입력값/결과를 초기화합니다."""
    word_list = st.session_state.word_list
    order = st.session_state.order_indices or list(range(len(word_list)))
    if not word_list or len(order) == 0:
        return
    # 순서 포인터를 이동하고, 표시 인덱스를 갱신
    st.session_state.current_order_pos = (st.session_state.current_order_pos + 1) % len(order)
    st.session_state.current_real_index = order[st.session_state.current_order_pos]
    st.session_state.user_input = ''
    st.session_state.last_result = None
    # 위젯 입력도 초기화
    if 'user_input' in st.session_state:
        st.session_state.user_input = ''


# 통계 계산
def compute_stats(word_list: list[dict]) -> dict:
    """총/완료/진행 중 개수를 계산합니다."""
    total = len(word_list)
    completed = 0
    for w in word_list:
        attempts = len(w['attempt_history'])
        if attempts > 0:
            completed += 1
    in_progress = total - completed
    return {
        'total': total,
        'completed': completed,
        'in_progress': in_progress,
    }


# 단어 상태 분류
def is_known_word(word_item: dict) -> bool:
    """아는 단어: 오답 0회이면서 정답(수동 포함) 1회 이상."""
    total_correct = word_item['correct_count'] + word_item['manual_correct_count']
    return word_item['wrong_count'] == 0 and total_correct > 0


def is_unknown_word(word_item: dict) -> bool:
    """모르는 단어: 정답 0회이면서 오답 1회 이상."""
    total_correct = word_item['correct_count'] + word_item['manual_correct_count']
    return total_correct == 0 and word_item['wrong_count'] > 0


# 통계/필터 섹션 렌더링
def render_stats_and_filters() -> None:
    """통계 패널과 필터링된 단어 목록을 표시합니다."""
    word_list = st.session_state.word_list
    stats = compute_stats(word_list)

    st.subheader('통계')
    c1, c2, c3 = st.columns(3)
    c1.metric('총 개수', stats['total'])
    c2.metric('완료 개수', stats['completed'])
    c3.metric('진행 중 개수', stats['in_progress'])

    st.subheader('단어 목록')
    filtered = word_list

    display_rows = []
    for w in filtered:
        display_rows.append({
            '단어': w['word'],
            '정답': w['meaning'],
            '처음 학습': w.get('first_learned_at', ''),
            '맞은 횟수': w['correct_count'],
            '오답 횟수': w['wrong_count'],
            '임의 정답': w['manual_correct_count'],
            '시도 수': len(w['attempt_history']),
        })

    st.dataframe(display_rows, use_container_width=True)


# 세션 리셋
def reset_session() -> None:
    """세션 상태를 초기화합니다."""
    st.session_state.word_list = load_default_dataset()
    st.session_state.current_index = 0
    st.session_state.user_input = ''
    st.session_state.last_result = None
    st.session_state.order_indices = list(range(len(st.session_state.word_list)))
    random.shuffle(st.session_state.order_indices)
    st.session_state.current_order_pos = 0
    st.session_state.current_real_index = st.session_state.order_indices[0] if st.session_state.order_indices else 0
    st.session_state.shuffled = True
    st.session_state.exam_mode = 'all'


def reshuffle_order() -> None:
    """단어 순서를 재셔플합니다."""
    total = len(st.session_state.word_list)
    st.session_state.order_indices = list(range(total))
    random.shuffle(st.session_state.order_indices)
    st.session_state.current_order_pos = 0
    st.session_state.current_real_index = st.session_state.order_indices[0] if st.session_state.order_indices else 0
    st.session_state.shuffled = True


def build_order_indices_for_mode() -> None:
    """현재 exam_mode에 맞는 순서 배열을 생성합니다."""
    words = st.session_state.word_list
    mode = st.session_state.exam_mode
    indices = list(range(len(words)))
    if mode == 'wrong_only':
        filtered = []
        for i in indices:
            w = words[i]
            total_correct = w['correct_count'] + w['manual_correct_count']
            if total_correct == 0 and w['wrong_count'] > 0:
                filtered.append(i)
        indices = filtered
    st.session_state.order_indices = indices
    random.shuffle(st.session_state.order_indices)
    st.session_state.current_order_pos = 0
    st.session_state.current_real_index = st.session_state.order_indices[0] if st.session_state.order_indices else 0


# 메인 함수
def main() -> None:
    """영어 단어 시험 웹페이지 메인 함수."""
    st.set_page_config(page_title='영어 단어 시험', layout='wide')
    st.title('영어 단어 시험')

    initialize_session_state()

    with st.sidebar:
        st.header('설정')
        st.markdown('시험 대상')
        mode_choice = st.radio('모드 선택', ['전체', '틀린 문제만'], index=0 if st.session_state.exam_mode=='all' else 1)
        new_mode = 'all' if mode_choice == '전체' else 'wrong_only'
        if new_mode != st.session_state.exam_mode:
            st.session_state.exam_mode = new_mode
            build_order_indices_for_mode()
            st.success('시험 대상을 반영했습니다.')
        uploaded = st.file_uploader('단어 CSV 업로드', type=['csv'])
        if uploaded is not None:
            try:
                new_list = load_csv_dataset(uploaded)
                st.session_state.word_list = new_list
                st.success(f'CSV에서 {len(new_list)}개 단어를 불러왔습니다.')
                st.session_state.current_index = 0
                st.session_state.user_input = ''
                st.session_state.last_result = None
                st.session_state.exam_mode = 'all'
                build_order_indices_for_mode()
                st.session_state.shuffled = True
            except Exception as e:
                st.error(f'CSV 로딩 실패: {e}')
        if st.button('셔플 다시'):
            build_order_indices_for_mode()
            st.success('단어 순서를 재셔플했습니다.')
        if st.button('세션 리셋'):
            reset_session()
            st.success('세션을 초기화했습니다.')

    tab_exam, tab_stats = st.tabs(['시험', '통계'])
    with tab_exam:
        render_quiz_ui()
    with tab_stats:
        render_stats_and_filters()


if __name__ == '__main__' :
main()
