import streamlit as st
import openai
import pandas as pd
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ===== 設定 =====
st.set_page_config(
    page_title="算数教員サポート CoPilot",
    page_icon="📐",
    layout="centered"
)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "1387017_004.pdf"
BRIDGE_DATA_PATH = "bridge_samples.json"

GRADE_INFO = {
    "小学1年": "Grade 1 (age 6-7): Numbers up to 100, addition and subtraction within 20. Use very simple words and concrete examples like fingers or physical objects.",
    "小学2年": "Grade 2 (age 7-8): Multiplication tables 2-9, addition and subtraction within 1000. Use arrays and equal groups as examples.",
    "小学3年": "Grade 3 (age 8-9): Division, basic fractions (1/2, 1/3), large numbers up to 10000. Connect new concepts to multiplication facts.",
    "小学4年": "Grade 4 (age 9-10): Multi-digit multiplication and division, fractions with same denominator, decimals, area of rectangles.",
    "小学5年": "Grade 5 (age 10-11): Fractions with different denominators, ratios, percentages, area of triangles and parallelograms, volume.",
    "小学6年": "Grade 6 (age 11-12): Ratios and rates, proportional relationships, area of circles, volume of cylinders, introduction to algebraic thinking.",
}

STRATEGIES = [
    "Ask a question",
    "Explain a concept",
    "Provide a hint",
    "Provide a worked example",
    "Affirm correct attempt",
    "Provide a similar problem",
]

# ===== テンプレートA：オリジナル日本語版 =====
ORIGINAL_TEMPLATE = """You are an experienced Japanese elementary school math teacher \
supporting a novice teacher (1-2 years of experience) who is conducting \
a 1-on-1 online tutoring session.

Student level: {grade_info}

Relevant curriculum context from Japanese national curriculum:
{curriculum}

The student is working on: {lesson_topic}.
The novice teacher needs expert guidance on how to respond to the \
student's current situation in a helpful and encouraging way.
Adjust your language complexity to match the student's grade level.
In your response, please {z}.
{c_h}
IMPORTANT: Write your response in Japanese.
teacher (最大1文、日本語で):"""

# ===== テンプレートC：Bridge論文テンプレート再現版 =====
# 出典：Wang et al. (2025) Section 3 / github.com/rosewang2008/tutor-copilot
# 論文のオリジナルテンプレート文言をそのまま使用
# ※ Think-Aloudデータ・学習済みモデルは含まない
BRIDGE_TEMPLATE = """You are an experienced elementary math teacher \
and you are going to respond to a student's mistake \
in a useful and caring way!!

The problem your student is solving is on topic: {lesson_topic}.
In your response, please {z}.
{c_h}
tutor (maximum one sentence):"""

# ===== テンプレートB：授業準備 =====
PREP_TEMPLATE = """You are an expert Japanese elementary school math educator \
with 20+ years of experience.
A novice teacher (1-2 years of experience) is preparing a lesson and needs \
your expert guidance.

Student level: {grade_info}

Relevant curriculum context from Japanese national curriculum:
{curriculum}

Lesson topic: {topic}

Please provide structured lesson preparation support in the following \
THREE sections.

---

## SECTION 1: Teaching Approaches (3 variations)
Provide exactly 3 distinct approaches to introduce or explain this topic.
Each approach should use a different method (e.g., visual, hands-on, \
story-based, connecting to prior knowledge).
For each approach:
- Approach name (short label)
- Core idea (1-2 sentences)
- Concrete example or activity

## SECTION 2: Guiding Questions (5 questions)
Provide exactly 5 questions the teacher can ask to guide student thinking \
WITHOUT giving away the answer.
Order them from simpler to more challenging.
For each question:
- The question itself
- What understanding it aims to reveal

## SECTION 3: Common Mistakes & How to Respond
Identify exactly 3 common mistakes students make with this topic.
CRITICAL: Each mistake's example, response, and explanation must be \
internally consistent and directly related to each other.
For each mistake:
- What the student does wrong (be specific to this topic)
- Example of a wrong answer (must match the mistake described)
- How to respond WITHOUT giving away the answer
- Why students make this mistake

---
Keep each section concise and immediately actionable for a novice teacher.
IMPORTANT: Write the entire response in Japanese. Use natural Japanese \
suitable for elementary school teachers in Japan."""

# ===== RAG：指導要領DB =====
@st.cache_resource
def build_vectorstore():
    if not os.path.exists(PDF_PATH):
        return None
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return FAISS.from_documents(docs, embeddings)

def search_curriculum(vectorstore, grade, topic, k=3):
    if vectorstore is None:
        return ""
    query = f"{grade} {topic} 算数 指導要領"
    results = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

# ===== Bridgeデータセット読み込み =====
@st.cache_resource
def load_bridge_samples():
    if not os.path.exists(BRIDGE_DATA_PATH):
        return []
    with open(BRIDGE_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def format_bridge_conversation(c_h_list):
    """Bridge形式の会話リストを文字列に変換"""
    return "\n".join(
        [f"{row['user']}: {row['text']}" for row in c_h_list]
    )

def bridge_to_display(c_h_list):
    """Bridge会話を表示用テキストに変換"""
    return "\n".join(
        [f"{row['user']}: {row['text']}" for row in c_h_list]
    )

# ===== 共通：AI生成 =====
def generate(prompt, max_tokens=150):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# ===== 匿名化・フォーマット =====
def deidentify(df, tutor_name, student_name):
    df = df.copy()
    df["text"] = df["text"].str.replace(
        tutor_name, "[TUTOR]", regex=False
    )
    df["text"] = df["text"].str.replace(
        student_name, "[STUDENT]", regex=False
    )
    df["user"] = df["user"].str.replace(
        tutor_name, "tutor", regex=False
    )
    df["user"] = df["user"].str.replace(
        student_name, "student", regex=False
    )
    return df

def format_conversation(df):
    return "\n".join(
        df.apply(lambda x: f"{x['user']}: {x['text']}", axis=1)
    )

def parse_conversation(conversation_input, tutor_name, student_name):
    rows = []
    for line in conversation_input.strip().split("\n"):
        if ": " in line:
            user, text = line.split(": ", 1)
            rows.append({"user": user.strip(), "text": text.strip()})
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df_anon = deidentify(df, tutor_name, student_name)
    c_h = format_conversation(df_anon)
    return df_anon, c_h

def parse_sections(text):
    sections = {"approaches": "", "questions": "", "mistakes": ""}
    if "SECTION 1" in text and "SECTION 2" in text and "SECTION 3" in text:
        parts = text.split("## SECTION")
        for part in parts[1:]:
            if part.startswith(" 1"):
                sections["approaches"] = part.replace(
                    "1: Teaching Approaches (3 variations)", ""
                ).strip()
            elif part.startswith(" 2"):
                sections["questions"] = part.replace(
                    "2: Guiding Questions (5 questions)", ""
                ).strip()
            elif part.startswith(" 3"):
                sections["mistakes"] = part.replace(
                    "3: Common Mistakes & How to Respond", ""
                ).strip()
    else:
        sections["approaches"] = text
    return sections

# ===== 初期化 =====
with st.spinner("指導要領データベースを準備しています..."):
    vectorstore = build_vectorstore()

bridge_samples = load_bridge_samples()

if vectorstore is None:
    st.warning(
        "⚠️ 指導要領PDFが見つかりません。PDFなしモードで動作します。"
    )

# ===== UI：ヘッダー =====
st.title("📐 算数教員サポート CoPilot")
st.caption(
    "教員向け　授業中リアルタイム支援 ＆ 授業準備アシスタント "
    "＆ Bridge論文データとの比較"
)
st.divider()

# ===== セッションステート初期化 =====
if "shared_grade_index" not in st.session_state:
    st.session_state.shared_grade_index = 4
if "shared_topic" not in st.session_state:
    st.session_state.shared_topic = ""
if "shared_tutor" not in st.session_state:
    st.session_state.shared_tutor = "Tanaka Sensei"
if "shared_student" not in st.session_state:
    st.session_state.shared_student = "Yuki"
if "shared_convo" not in st.session_state:
    st.session_state.shared_convo = ""

# ===== タブ =====
tab_a, tab_b, tab_c = st.tabs([
    "💬 授業中サポート（日本語版）",
    "📝 授業準備アシスタント",
    "🔬 Bridge論文データとの比較",
])

# ==========================================
# タブA：授業中サポート（日本語・指導要領版）
# ==========================================
with tab_a:
    st.markdown(
        "#### 授業中に児童がつまずいたとき、次の一手を提案します"
    )
    st.caption("入力内容はタブC（Bridge比較版）と共有されます")
    st.markdown("")

    grade_a = st.radio(
        "🎓 学年",
        options=list(GRADE_INFO.keys()),
        horizontal=True,
        index=st.session_state.shared_grade_index,
        key="grade_a"
    )
    st.session_state.shared_grade_index = (
        list(GRADE_INFO.keys()).index(grade_a)
    )

    lesson_topic_a = st.text_input(
        "📚 単元・テーマ",
        placeholder="例：わり算の導入、分数のたし算、割合と百分率",
        key="topic_a",
        value=st.session_state.shared_topic
    )
    st.session_state.shared_topic = lesson_topic_a

    st.markdown("**💬 会話を入力**")
    col1, col2 = st.columns(2)
    with col1:
        tutor_name_a = st.text_input(
            "チューターの名前",
            value=st.session_state.shared_tutor,
            key="tutor_a"
        )
        st.session_state.shared_tutor = tutor_name_a
    with col2:
        student_name_a = st.text_input(
            "児童の名前",
            value=st.session_state.shared_student,
            key="student_a"
        )
        st.session_state.shared_student = student_name_a

    conversation_a = st.text_area(
        "会話（「名前: 発言」の形式で入力）",
        height=180,
        placeholder=(
            "Tanaka Sensei: What is 1/2 + 1/3?\n"
            "Yuki: Is it 2/5?"
        ),
        key="convo_a",
        value=st.session_state.shared_convo
    )
    st.session_state.shared_convo = conversation_a

    if st.button(
        "💡 日本語版で提案を生成する",
        type="primary",
        disabled=not lesson_topic_a,
        key="btn_a"
    ):
        with st.spinner(
            "AIが提案を生成しています（日本語・指導要領参照）..."
        ):
            _, c_h = parse_conversation(
                conversation_a, tutor_name_a, student_name_a
            )
            if c_h is None:
                st.error("会話を入力してください")
                st.stop()

            curriculum = search_curriculum(
                vectorstore, grade_a, lesson_topic_a
            )
            responses_a = []
            for strategy in STRATEGIES:
                prompt = ORIGINAL_TEMPLATE.format(
                    grade_info=GRADE_INFO[grade_a],
                    curriculum=(
                        curriculum if curriculum else "Not available."
                    ),
                    lesson_topic=lesson_topic_a,
                    z=strategy.lower(),
                    c_h=c_h
                )
                responses_a.append(generate(prompt, max_tokens=120))

        st.divider()
        st.markdown(
            f"### 💜 Let's help the student!"
            f"　（{grade_a}・{lesson_topic_a}）"
        )
        st.caption(
            "🇯🇵 日本語版：学年別対応・文科省指導要領参照・日本語出力"
        )

        if curriculum:
            with st.expander("📖 参照した指導要領の内容"):
                st.caption(curriculum)

        cols = st.columns(3)
        for i, (strategy, response) in enumerate(
            zip(STRATEGIES, responses_a)
        ):
            with cols[i % 3]:
                st.markdown(f"**{strategy}**")
                st.info(response)

        st.divider()
        st.caption(
            "※ 提案は参考情報です。"
            "実際の授業では教員の判断・児童の実態に合わせて調整してください。"
        )

# ==========================================
# タブB：授業準備アシスタント
# ==========================================
with tab_b:
    st.markdown(
        "#### 単元・テーマを入力すると指導アイデアを3つの視点で生成します"
    )
    st.markdown("")

    grade_b = st.radio(
        "🎓 学年",
        options=list(GRADE_INFO.keys()),
        horizontal=True,
        index=4,
        key="grade_b"
    )

    prep_topic = st.text_input(
        "📝 単元・テーマ",
        placeholder="例：かけ算の導入、分母の違う分数のたし算、速さと時間",
        key="topic_b"
    )

    if st.button(
        "📋 授業アイデアを生成する",
        type="primary",
        disabled=not prep_topic,
        key="btn_b"
    ):
        with st.spinner("専門家レベルの指導アイデアを生成しています..."):
            curriculum = search_curriculum(
                vectorstore, grade_b, prep_topic
            )
            prompt = PREP_TEMPLATE.format(
                grade_info=GRADE_INFO[grade_b],
                curriculum=(
                    curriculum if curriculum else "Not available."
                ),
                topic=prep_topic
            )
            raw = generate(prompt, max_tokens=1500)
            sections = parse_sections(raw)

        st.divider()
        st.markdown(f"### 📐 {grade_b}・{prep_topic}　指導サポート")

        if curriculum:
            with st.expander("📖 参照した指導要領の内容"):
                st.caption(curriculum)

        st.markdown("#### 🎯 指導アプローチ（3つのバリエーション）")
        st.info(sections["approaches"])
        st.markdown("#### 💬 発問例（考えさせる問いかけ5選）")
        st.success(sections["questions"])
        st.markdown("#### ⚠️ よくあるつまずきと対処法")
        st.warning(sections["mistakes"])

        st.divider()
        st.caption(
            "※ 提案は参考情報です。"
            "実際の授業では教員の判断で調整してください。"
        )

# ==========================================
# タブC：Bridge論文データとの比較
# ==========================================
with tab_c:
    st.markdown("#### 同じ会話・同じテンプレートで2つの実装を比較します")
    st.markdown("")

    # 2種類の比較モードを選択
    compare_mode = st.radio(
        "比較モードを選択",
        options=[
            "🔄 タブAの入力で比較（自分で会話を入力）",
            "📂 Bridge論文データセットのサンプルで比較"
        ],
        key="compare_mode"
    )

    st.divider()

    # ===== モード①：タブAの入力を使う =====
    if compare_mode == "🔄 タブAの入力で比較（自分で会話を入力）":

        col_desc1, col_desc2 = st.columns(2)
        with col_desc1:
            st.markdown(
                "**🇯🇵 日本語版（本プロトタイプ）**\n"
                "- 文科省指導要領PDF参照\n"
                "- 学年別対応\n"
                "- 日本語出力"
            )
        with col_desc2:
            st.markdown(
                "**📄 Bridge論文テンプレート再現版**\n"
                "- Wang et al. (2025) Section 3\n"
                "- 学年・指導要領参照なし\n"
                "- 英語出力（論文に忠実）"
            )

        grade_c = list(GRADE_INFO.keys())[
            st.session_state.shared_grade_index
        ]
        topic_c = st.session_state.shared_topic
        tutor_c = st.session_state.shared_tutor
        student_c = st.session_state.shared_student
        convo_c = st.session_state.shared_convo

        with st.expander(
            "📋 タブAから引き継いだ入力内容", expanded=True
        ):
            st.markdown(f"**学年：** {grade_c}")
            st.markdown(
                f"**単元：** {topic_c if topic_c else '（未入力）'}"
            )
            st.markdown(
                f"**チューター：** {tutor_c}　　"
                f"**児童：** {student_c}"
            )
            if convo_c:
                st.text(convo_c)
            else:
                st.caption(
                    "会話が未入力です。タブAで入力してください。"
                )

        if st.button(
            "🔬 2つの実装を並べて比較する",
            type="primary",
            disabled=not (topic_c and convo_c),
            key="btn_c1"
        ):
            with st.spinner("2つの実装で提案を生成しています..."):
                _, c_h = parse_conversation(
                    convo_c, tutor_c, student_c
                )
                if c_h is None:
                    st.error(
                        "タブAで会話を入力してから比較を実行してください"
                    )
                    st.stop()

                curriculum = search_curriculum(
                    vectorstore, grade_c, topic_c
                )
                responses_orig, responses_bridge = [], []

                for strategy in STRATEGIES:
                    prompt_orig = ORIGINAL_TEMPLATE.format(
                        grade_info=GRADE_INFO[grade_c],
                        curriculum=(
                            curriculum if curriculum
                            else "Not available."
                        ),
                        lesson_topic=topic_c,
                        z=strategy.lower(),
                        c_h=c_h
                    )
                    responses_orig.append(
                        generate(prompt_orig, max_tokens=120)
                    )

                    prompt_bridge = BRIDGE_TEMPLATE.format(
                        lesson_topic=topic_c,
                        z=strategy.lower(),
                        c_h=c_h
                    )
                    responses_bridge.append(
                        generate(prompt_bridge, max_tokens=120)
                    )

            _show_comparison(
                grade_c, topic_c, curriculum,
                responses_orig, responses_bridge
            )

    # ===== モード②：Bridge論文データセットを使う =====
    else:
        st.markdown(
            "##### 📂 Bridge論文データセット（Wang et al., 2024）"
        )
        st.caption(
            "出典：github.com/rosewang2008/bridge / dataset/test.json　"
            "米国の数学授業の実際の会話データ（英語）"
        )

        if not bridge_samples:
            st.error(
                "bridge_samples.jsonが見つかりません。"
                "GitHubに追加してください。"
            )
        else:
            # サンプル選択
            sample_options = {
                f"サンプル {s['id']}：{s['lesson_topic']}": s
                for s in bridge_samples
            }
            selected_label = st.selectbox(
                "使用するサンプルを選択",
                options=list(sample_options.keys()),
                key="bridge_sample_select"
            )
            selected = sample_options[selected_label]

            # 選択されたサンプルの会話を表示
            with st.expander(
                "📋 選択したBridgeデータの会話内容", expanded=True
            ):
                st.caption(
                    f"**lesson_topic：** {selected['lesson_topic']}"
                )
                st.caption(f"**データ出典：** {selected['note']}")
                st.text(bridge_to_display(selected["c_h"]))

            # 日本語版用の学年選択
            st.markdown(
                "🇯🇵 **日本語版に適用する学年を選択**"
                "（Bridge版は学年指定なし）"
            )
            grade_c2 = st.radio(
                "学年",
                options=list(GRADE_INFO.keys()),
                horizontal=True,
                index=4,
                key="grade_c2"
            )

            if st.button(
                "🔬 Bridgeデータで2つの実装を比較する",
                type="primary",
                key="btn_c2"
            ):
                with st.spinner(
                    "Bridge論文データで2つの実装を比較しています..."
                ):
                    # Bridge会話データをフォーマット
                    c_h_bridge = format_bridge_conversation(
                        selected["c_h"]
                    )
                    topic_bridge = selected["lesson_topic"]

                    curriculum = search_curriculum(
                        vectorstore, grade_c2, topic_bridge
                    )
                    responses_orig, responses_bridge = [], []

                    for strategy in STRATEGIES:
                        # 日本語版（指導要領・学年あり）
                        prompt_orig = ORIGINAL_TEMPLATE.format(
                            grade_info=GRADE_INFO[grade_c2],
                            curriculum=(
                                curriculum if curriculum
                                else "Not available."
                            ),
                            lesson_topic=topic_bridge,
                            z=strategy.lower(),
                            c_h=c_h_bridge
                        )
                        responses_orig.append(
                            generate(prompt_orig, max_tokens=120)
                        )

                        # Bridge論文テンプレート版
                        prompt_bridge = BRIDGE_TEMPLATE.format(
                            lesson_topic=topic_bridge,
                            z=strategy.lower(),
                            c_h=c_h_bridge
                        )
                        responses_bridge.append(
                            generate(prompt_bridge, max_tokens=120)
                        )

                _show_comparison(
                    grade_c2, topic_bridge, curriculum,
                    responses_orig, responses_bridge,
                    is_bridge_data=True
                )


def _show_comparison(
    grade, topic, curriculum,
    responses_orig, responses_bridge,
    is_bridge_data=False
):
    """比較結果の表示（共通関数）"""
    st.divider()
    st.markdown(f"### 🔬 比較結果　{grade}・{topic}")

    if is_bridge_data:
        st.info(
            "📂 Bridge論文データセット（米国・英語）を使用した比較です。"
            "日本語版は日本の学習指導要領を参照しますが、"
            "会話は英語のままです。"
        )

    if curriculum:
        with st.expander("📖 日本語版が参照した指導要領の内容"):
            st.caption(curriculum)

    for i, strategy in enumerate(STRATEGIES):
        st.markdown(f"---\n#### {strategy}")
        col_orig, col_bridge = st.columns(2)

        with col_orig:
            st.markdown(
                "🇯🇵 **日本語版**"
                "（指導要領参照・学年別・日本語出力）"
            )
            st.info(responses_orig[i])

        with col_bridge:
            st.markdown(
                "📄 **Bridge論文テンプレート再現版**"
                "（Wang et al., 2025）"
            )
            st.warning(responses_bridge[i])

    st.divider()
    st.markdown("#### 比較のポイント")
    st.markdown(
        "| 観点 | 確認すること |\n"
        "|------|------------|\n"
        "| **言語** | 日本語版は日本語か・Bridge版は英語か |\n"
        "| **学年適合性** | 日本語版の言葉レベルは学年に合っているか |\n"
        "| **指導要領の反映** | 日本語版は指導要領の内容を踏まえているか |\n"
        "| **教育的質** | どちらが「考えさせる」提案になっているか |\n"
        "| **文化的文脈** | 米国データに日本版がどう対応しているか |"
    )
    st.caption(
        "出典（Bridge版テンプレート・データ）："
        'Wang et al. (2025) "Tutor CoPilot: A Human-AI Approach '
        'for Scaling Real-Time Expertise" arXiv:2410.03017v2 / '
        "Wang et al. (2024) NAACL「Bridge」/ "
        "github.com/rosewang2008/bridge"
    )
