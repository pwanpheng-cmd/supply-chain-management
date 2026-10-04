# Feedback: การทดสอบ SCM Data Chat (QA)

วันที่: 2026-10-04 · สถานะ: **ยังไม่ได้ทดสอบ** (ไม่มีผล Accuracy / Security / Performance)

## สรุป
- งาน QA ตามขั้นที่ 1-6 ยังไม่ได้ดำเนินการ เพราะ session ที่ได้รับมอบหมายรันบน Linux cloud container ไม่ใช่เครื่อง Windows ของผู้ใช้
- สิ่งที่ตรวจพบใน container: ไม่มี path `C:\Users\Admin\...`, ไม่มี `chat_server.py`, `scm_data_center.sqlite`, `ollama`, `py`
- repo `pwanpheng-cmd/supply-chain-management` มีเพียง `SCM_App.py`, `README`, `requirements.txt` (ไม่มีโฟลเดอร์ `chatbot` หรือ Dashboard)
- ไม่มีการแก้ไขไฟล์โค้ด / ฐานข้อมูล / config ใดๆ ในการตรวจนี้

## สิ่งที่ต้องทำต่อ (บนเครื่อง Windows)
1. เปิด Claude Code แบบ local โดยตั้ง working directory เป็น
   `C:\Users\Admin\Documents\AI Project\App19_SCM_Data_Center\SCM_Report\Dashboard SC Executive`
2. วาง prompt QA เดิมทั้งหมด (ขั้นที่ 1-6 + รายงาน)
3. ถ้า `ollama list` ไม่มี `qwen2.5:7b` ให้ยืนยันก่อน `ollama pull` (~4.7 GB)
4. ขั้นที่ 6 (Dashboard) ต้องมีเครื่องมือเปิดเบราว์เซอร์ ไม่เช่นนั้นจะถูกข้าม
5. เมื่อได้ `chatbot\tests\TEST_REPORT_<YYYYMMDD>.md` ให้ส่งกลับมาเพื่อทบทวน logic และเสนอกติกาเพิ่มใน `knowledge.md` (รออนุมัติก่อนแก้ไฟล์)

## ข้อสังเกตล่วงหน้า (จาก prompt ไม่ใช่ผลทดสอบ) — จุดเสี่ยงที่ควรจับตา
- ข้อ 3, 5, 9: เงื่อนไขธุรกิจเยอะ (ตัด Done/Closed/Cancel, `-Del`, `is_active=1`, สูตร `actual_qty*order_cost/order_qty`) LLM 7B มักลืมเงื่อนไขเหล่านี้ → ควรมีตัวอย่าง SQL ใน knowledge.md
- ข้อ 10: คำถามต่อเนื่องขึ้นกับการส่ง `history` ที่ถูกต้อง
- Security: ควรยืนยันว่า `/api/run` ใช้ read-only connection (`mode=ro`) ไม่ใช่แค่ตรวจคำสั่งด้วย keyword
- Performance: โมเดล 7B + DB 2 GB บนเครื่องที่ไม่มี GPU อาจเกิน 90 วินาที ควรบันทึกสเปกเครื่องควบคู่
