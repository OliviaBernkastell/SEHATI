import streamlit as st

def main():
    st.set_page_config(
        page_title="SEHATI - Beranda",
        page_icon="🏠",
    )

    st.title("Selamat Datang di SEHATI")
    st.header("Solusi Energi dan Hidup Sehat Terkini")

    st.write("""
    **SEHATI** adalah asisten pribadi Anda untuk menghitung kebutuhan kalori harian dan memberikan rekomendasi makanan yang dipersonalisasi.

    - Gunakan sidebar untuk menavigasi fitur-fitur aplikasi.
    - Mulailah dengan pergi ke halaman **Aplikasi SEHATI** untuk memasukkan data pribadi Anda.
    - Setelah menghitung kebutuhan kalori Anda, Anda dapat menerima rekomendasi makanan yang dipersonalisasi.
    - Anda juga dapat mengobrol dengan SEHATI untuk pertanyaan seputar kesehatan dan nutrisi.

    Kami berharap SEHATI dapat membantu Anda dalam perjalanan menuju gaya hidup yang lebih sehat!
    """)

    # Menampilkan gambar (opsional)
    st.image("chatbot_profile.png", use_column_width=True)

if __name__ == "__main__":
    main()
