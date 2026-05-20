import streamlit as st
import openai
import pandas as pd
import os
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
PDF_PATH = "1387017_004.pdf"  # ← GitHubにアップロードしたPDFのファイル名に合わせる

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

# ===== RAG：指導要領DBの構築（初回のみ）=====
@st.cache_resource
def build_vectorstore():
    if not os.path.exists(PDF_PATH):
        return None
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def search_curriculum(vectorstore, grade, topic, k=3):
    if vectorstore is None:
        return ""
    query = f"{grade} {topic} 算数 指導要領"
    results = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

# ===== 共通：AI生成 =====
def generate(prompt, max_tokens=150):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# ===== タブA用：匿名化・フォーマット =====
def deidentify(df, tutor_name, student_name):
    df = df.copy()
    df["text"] = df["text"].str.replace(tutor_name, "[TUTOR]", regex=False)
    df["text"] = df["text"].str.replace(student_name, "[STUDENT]", regex=False)
    df["user"] = df["user"].str.replace(tutor_name, "tutor", regex=False)
    df["user"] = df["user"].str.replace(student_name, "student", regex=False)
    return df

def format_conversation(df):
    return "\n".join(df.apply(lambda x: f"{x['user']}: {x['text']}", axis=1))

# ===== タブA用：テンプレート =====
REALTIME_TEMPLATE = """You are an experienced Japanese elementary school math teacher \
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

# ===== タブB用：テンプレート =====
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
CRITICAL: Each mistake's example, response, and explanation must be 
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

# ===== RAG初期化 =====
with st.spinner("指導要領データベースを準備しています..."):
    vectorstore = build_vectorstore()

if vectorstore is None:
    st.warning("⚠️ 指導要領PDFが見つかりません。PDFなしモードで動作します。")

# ===== UI：ヘッダー =====
st.title("📐 算数教員サポート CoPilot")
st.caption("教員向け　授業中リアルタイム支援 ＆ 授業準備アシスタント")
st.divider()

# ===== タブ =====
tab_a, tab_b = st.tabs([
    "💬 授業中サポート（リアルタイム）",
    "📝 授業準備アシスタント"
])

# ==========================================
# タブA：授業中リアルタイムサポート
# ==========================================
with tab_a:
    st.markdown("#### 授業中に児童がつまずいたとき、次の一手を提案します")
    st.markdown("")

    grade_a = st.radio(
        "🎓 学年",
        options=list(GRADE_INFO.keys()),
        horizontal=True,
        index=4,
        key="grade_a"
    )

    lesson_topic = st.text_input(
        "📚 単元・テーマ",
        placeholder="例：わり算の導入、分数のたし算、割合と百分率",
        key="topic_a"
    )

    st.markdown("**💬 会話を入力**")
    col1, col2 = st.columns(2)
    with col1:
        tutor_name = st.text_input(
            "チューターの名前", value="Tanaka Sensei", key="tutor"
        )
    with col2:
        student_name = st.text_input(
            "生徒の名前", value="Yuki", key="student"
        )

    conversation_input = st.text_area(
        "会話（「名前: 発言」の形式で入力）",
        height=180,
        placeholder="Tanaka Sensei: What is 1/2 + 1/3?\nYuki: Is it 2/5?",
        key="convo"
    )

    if st.button("💡 提案を生成する", type="primary",
                 disabled=not lesson_topic, key="btn_a"):
        with st.spinner("AIが提案を生成しています..."):

            rows = []
            for line in conversation_input.strip().split("\n"):
                if ": " in line:
                    user, text = line.split(": ", 1)
                    rows.append({"user": user.strip(), "text": text.strip()})

            if not rows:
                st.error("会話を入力してください")
                st.stop()

            df = pd.DataFrame(rows)
            df_anon = deidentify(df, tutor_name, student_name)
            c_h = format_conversation(df_anon)

            # 指導要領から関連箇所を取得
            curriculum = search_curriculum(
                vectorstore, grade_a, lesson_topic
            )

            responses = []
            for strategy in STRATEGIES:
                prompt = REALTIME_TEMPLATE.format(
                    grade_info=GRADE_INFO[grade_a],
                    curriculum=curriculum if curriculum else "Not available.",
                    lesson_topic=lesson_topic,
                    z=strategy.lower(),
                    c_h=c_h
                )
                responses.append(generate(prompt, max_tokens=100))

        st.divider()
        st.markdown(
            f"### 💜 Let's help the student!　（{grade_a}・{lesson_topic}）"
        )

        if curriculum:
            with st.expander("📖 参照した指導要領の内容"):
                st.caption(curriculum)

        cols = st.columns(3)
        for i, (strategy, response) in enumerate(zip(STRATEGIES, responses)):
            with cols[i % 3]:
                st.markdown(f"**{strategy}**")
                st.info(response)

        st.divider()
        st.caption(
            "※ 提案は参考情報です。実際の授業では児童の実態に合わせて調整してください。"
        )

# ==========================================
# タブB：授業準備アシスタント
# ==========================================
with tab_b:
    st.markdown("#### 単元・テーマを入力すると指導アイデアを3つの視点で生成します")
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

    if st.button("📋 授業アイデアを生成する", type="primary",
                 disabled=not prep_topic, key="btn_b"):
        with st.spinner("指導アイデアを生成しています..."):

            # 指導要領から関連箇所を取得
            curriculum = search_curriculum(
                vectorstore, grade_b, prep_topic
            )

            prompt = PREP_TEMPLATE.format(
                grade_info=GRADE_INFO[grade_b],
                curriculum=curriculum if curriculum else "Not available.",
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
            "※ 提案は参考情報です。実際の授業では児童の実態に合わせて調整してください。"
        )
