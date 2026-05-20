import streamlit as st
import openai
import pandas as pd

# ===== 設定 =====
st.set_page_config(
    page_title="Tutor CoPilot",
    page_icon="🌟",
    layout="centered"
)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

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

TEMPLATE = """You are an experienced Japanese elementary school math teacher \
supporting a novice teacher (1-2 years of experience) who is conducting \
a 1-on-1 online tutoring session.

Student level: {grade_info}

The student is working on: {lesson_topic}.
The novice teacher needs expert guidance on how to respond to the \
student's current situation in a helpful and encouraging way.
Adjust your language complexity to match the student's grade level.
In your response, please {z}.
{c_h}
teacher (maximum one sentence):"""

# ===== 匿名化 =====
def deidentify(df, tutor_name, student_name):
    df = df.copy()
    df["text"] = df["text"].str.replace(tutor_name, "[TUTOR]", regex=False)
    df["text"] = df["text"].str.replace(student_name, "[STUDENT]", regex=False)
    df["user"] = df["user"].str.replace(tutor_name, "tutor", regex=False)
    df["user"] = df["user"].str.replace(student_name, "student", regex=False)
    return df

# ===== 会話フォーマット =====
def format_conversation(df):
    return "\n".join(df.apply(lambda x: f"{x['user']}: {x['text']}", axis=1))

# ===== AI生成 =====
def generate(prompt):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content

# ===== UI =====
st.title("🌟 Tutor CoPilot")
st.caption("算数　教員リアルタイムサポート")
st.divider()

# 学年選択
grade = st.radio(
    "🎓 学年を選択",
    options=list(GRADE_INFO.keys()),
    horizontal=True,
    index=4  # デフォルト：小学5年
)

# 単元入力
lesson_topic = st.text_input(
    "📚 単元・テーマ",
    placeholder="例：わり算の導入、分数のたし算、割合と百分率"
)

st.divider()
st.markdown("**💬 会話を入力**")

col1, col2 = st.columns(2)
with col1:
    tutor_name = st.text_input("チューターの名前", value="Tanaka Sensei")
with col2:
    student_name = st.text_input("生徒の名前", value="Yuki")

conversation_input = st.text_area(
    "会話（「名前: 発言」の形式で入力）",
    height=200,
    placeholder="""例：
Tanaka Sensei: Today we are working on fractions. Are you ready?
Yuki: Yes!
Tanaka Sensei: What is 1/2 + 1/3?
Yuki: 2/5?"""
)

if st.button("💡 提案を生成する", type="primary", disabled=not (topic := lesson_topic)):
    with st.spinner("AIが提案を生成しています..."):

        # 会話をDataFrameに変換
        rows = []
        for line in conversation_input.strip().split("\n"):
            if ": " in line:
                user, text = line.split(": ", 1)
                rows.append({"user": user.strip(), "text": text.strip()})

        if not rows:
            st.error("会話を入力してください")
            st.stop()

        df = pd.DataFrame(rows)

        # 匿名化
        df_anon = deidentify(df, tutor_name, student_name)
        c_h = format_conversation(df_anon)

        # 各ストラテジーで生成
        responses = []
        for strategy in STRATEGIES:
            prompt = TEMPLATE.format(
                grade_info=GRADE_INFO[grade],
                lesson_topic=lesson_topic,
                z=strategy.lower(),
                c_h=c_h
            )
            response = generate(prompt)
            responses.append(response)

    # パネル表示
    st.divider()
    st.markdown(f"### 💜 Let's help the student!　（{grade}・{lesson_topic}）")

    cols = st.columns(3)
    for i, (strategy, response) in enumerate(zip(STRATEGIES, responses)):
        with cols[i % 3]:
            st.markdown(f"**{strategy}**")
            st.info(response)
            st.markdown("")

    st.divider()
    st.caption("※ 提案は参考情報です。実際の授業では児童・生徒の実態に合わせて調整してください。")
