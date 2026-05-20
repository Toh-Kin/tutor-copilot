import streamlit as st
import openai

# ===== 設定 =====
st.set_page_config(
    page_title="算数授業準備アシスタント",
    page_icon="📐",
    layout="centered"
)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

GRADE_INFO = {
    "小学1年": "Grade 1 (age 6-7): Numbers up to 100, addition and subtraction within 20. Students need very concrete, hands-on examples using fingers or physical objects. Use extremely simple language.",
    "小学2年": "Grade 2 (age 7-8): Multiplication tables 2-9, addition/subtraction within 1000. Students understand equal groups and arrays. Keep language simple and use familiar objects.",
    "小学3年": "Grade 3 (age 8-9): Division, basic fractions (1/2, 1/3), large numbers up to 10000. Connect new concepts to previously learned multiplication facts.",
    "小学4年": "Grade 4 (age 9-10): Multi-digit multiplication/division, fractions with same denominator, decimals, area of rectangles. Students can handle more abstract thinking.",
    "小学5年": "Grade 5 (age 10-11): Fractions with different denominators, ratios, percentages, area of triangles and parallelograms, volume. Students can engage with proportional reasoning.",
    "小学6年": "Grade 6 (age 11-12): Ratios and rates, proportional relationships, area of circles, volume of cylinders, introduction to algebraic thinking.",
}

TEMPLATE = """You are an expert Japanese elementary school math educator with 20+ years of experience.
A novice teacher (1-2 years of experience) is preparing a lesson and needs your expert guidance.

Student level: {grade_info}

Lesson topic: {topic}

Please provide structured lesson preparation support in the following THREE sections.
Write all teacher guidance in English, but example student dialogue can include Japanese math terms where natural.

---

## SECTION 1: Teaching Approaches (3 variations)
Provide exactly 3 distinct approaches to introduce or explain this topic.
Each approach should use a different method (e.g., visual, hands-on, story-based, connecting to prior knowledge).
For each approach:
- Approach name (short label)
- Core idea (1-2 sentences)
- Concrete example or activity

## SECTION 2: Guiding Questions (5 questions)
Provide exactly 5 questions the teacher can ask to guide student thinking WITHOUT giving away the answer.
These should follow the Socratic method - helping students discover the concept themselves.
Order them from simpler to more challenging.
For each question:
- The question itself
- What understanding it aims to reveal

## SECTION 3: Common Mistakes & How to Respond
Identify exactly 3 common mistakes or misconceptions students make with this topic.
For each mistake:
- Mistake description (what the student does wrong)
- Example of wrong answer a student might give
- How to respond WITHOUT giving away the answer (use a guiding question or hint)
- Why students make this mistake (brief explanation)

---
Keep each section concise and immediately actionable for a novice teacher."""

def generate_lesson_prep(grade, topic):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = TEMPLATE.format(
        grade_info=GRADE_INFO[grade],
        topic=topic
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return response.choices[0].message.content

def parse_sections(text):
    sections = {"approaches": "", "questions": "", "mistakes": ""}
    if "SECTION 1" in text and "SECTION 2" in text:
        parts = text.split("## SECTION")
        for part in parts[1:]:
            if part.startswith(" 1"):
                sections["approaches"] = part.split("## SECTION")[0].strip()
            elif part.startswith(" 2"):
                sections["questions"] = part.split("## SECTION")[0].strip()
            elif part.startswith(" 3"):
                sections["mistakes"] = part.strip()
    else:
        sections["approaches"] = text
    return sections

# ===== UI =====
st.title("📐 算数授業準備アシスタント")
st.caption("教員向け　指導アイデア・発問・つまずき対策を一括生成")
st.divider()

col1, col2 = st.columns([1, 2])
with col1:
    grade = st.selectbox(
        "🎓 学年",
        options=list(GRADE_INFO.keys())
    )
with col2:
    topic = st.text_input(
        "📝 単元・テーマ",
        placeholder="例：わり算の導入、分数のたし算、割合と百分率"
    )

st.divider()

if st.button("💡 授業アイデアを生成する", type="primary", disabled=not topic):
    with st.spinner("指導アイデアを生成しています..."):
        raw = generate_lesson_prep(grade, topic)
        sections = parse_sections(raw)

    # Section 1
    st.markdown("### 🎯 指導アプローチ（3つのバリエーション）")
    st.info(sections["approaches"].replace("1: Teaching Approaches (3 variations)", "").strip())

    st.divider()

    # Section 2
    st.markdown("### 💬 発問例（考えさせる問いかけ5選）")
    st.success(sections["questions"].replace("2: Guiding Questions (5 questions)", "").strip())

    st.divider()

    # Section 3
    st.markdown("### ⚠️ よくあるつまずきと対処法")
    st.warning(sections["mistakes"].replace("3: Common Mistakes & How to Respond", "").strip())

    st.divider()
    st.caption("※ 提案は参考情報です。実際の授業では児童の実態に合わせて調整してください。")
