import streamlit as st
import openai
import pandas as pd

# ===== 設定 =====
st.set_page_config(page_title="Tutor CoPilot", page_icon="🌟", layout="centered")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

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
a 1-on-1 online tutoring session with a 5th grade student.

The student is working on: {lesson_topic}.
The novice teacher needs expert guidance on how to respond to the \
student's current situation in a helpful and encouraging way.
In your response, please {z}.
{c_h}
teacher (maximum one sentence):"""

# ===== 匿名化 =====
def deidentify(df, tutor_name, student_name):
    processor = TextPreprocessor()
    known_names = [tutor_name, student_name]
    df = processor.anonymize_known_names(
        df=df, text_column="text", names=known_names,
        replacement_names=["[TUTOR]", "[STUDENT]"]
    )
    df = processor.anonymize_known_names(
        df=df, text_column="user", names=known_names,
        replacement_names=["tutor", "student"]
    )
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
st.caption("算数・小学5年生　若手教員サポートデモ")
st.divider()

lesson_topic = st.text_input(
    "📚 授業トピック（英語）",
    value="Ratios and percentages using fractions and decimals (Grade 5)"
)

st.markdown("**💬 会話を入力**")

tutor_name   = st.text_input("チューターの名前", value="Tanaka Sensei")
student_name = st.text_input("生徒の名前",       value="Yuki")

conversation_input = st.text_area(
    "会話（「名前: 発言」の形式で入力）",
    height=200,
    value="""Tanaka Sensei: Good morning! Today we are working on ratios. Are you ready?
Yuki: Yes!
Tanaka Sensei: In a class of 30 students, 18 like soccer. What is the ratio of soccer fans to the whole class?
Yuki: Umm... is it 18 divided by 30?
Tanaka Sensei: That is a good start! Can you calculate that?
Yuki: 0.06?"""
)

if st.button("💡 提案を生成する", type="primary"):
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

        responses = []
        for strategy in STRATEGIES:
            prompt = TEMPLATE.format(
                lesson_topic=lesson_topic,
                z=strategy.lower(),
                c_h=c_h
            )
            response = generate(prompt)
            responses.append(response)

    st.divider()
    st.markdown("### 💜 Let's help the student!")

    cols = st.columns(3)
    for i, (strategy, response) in enumerate(zip(STRATEGIES, responses)):
        with cols[i % 3]:
            st.markdown(f"**{strategy}**")
            st.info(response)
