import os, inspect
os.chdir(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import customtkinter as ctk
from huggingface_hub import HfApi, scan_cache_dir
from tkinter import messagebox
import tkinter as tk
import os

# 로그인 상태를 저장할 파일 경로 설정
LOGIN_FILE = "login_token.txt"

# Hugging Face API 인스턴스 생성
api = HfApi()

# 로그인 상태 복원
def load_login_token():
    if os.path.exists(LOGIN_FILE):
        with open(LOGIN_FILE, "r") as f:
            token = f.read().strip()
            api.set_access_token(token)
            return token
    return None

# 로그인 토큰 저장
def save_login_token(token):
    with open(LOGIN_FILE, "w") as f:
        f.write(token)

# 로그인 토큰 삭제
def delete_login_token():
    if os.path.exists(LOGIN_FILE):
        os.remove(LOGIN_FILE)

# GUI 초기 설정
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# 메인 윈도우 생성
app = ctk.CTk()
app.title("Hugging Face 캐시 및 사용자 관리")
app.geometry("900x600")  # 가로 크기 증가

# 캐시 정보 스캔 및 화면에 표시하는 기능
def scan_cache():
    global cache_info, revisions, sort_orders
    cache_info = scan_cache_dir()

    # 캐시 데이터 수집
    revisions = []
    for repo in cache_info.repos:
        for revision in repo.revisions:
            rev_info = {
                "repo_id": repo.repo_id,
                "revision": revision.commit_hash,
                "size": revision.size_on_disk,
                "last_modified": revision.last_modified,
            }
            revisions.append(rev_info)
    
    # 각 열의 초기 정렬 순서 (True: 오름차순, False: 내림차순)
    sort_orders = {
        "repo_id": True,
        "revision": True,
        "size": False,
        "last_modified": True,
    }

    display_cache_items()

# 선택된 항목의 개수와 총 용량을 합산하여 표시하는 함수
def update_selection_summary():
    selected_count = 0
    total_size = 0
    for var, rev in checkboxes:
        if var.get():
            selected_count += 1
            total_size += rev['size']

    label_selection_summary.configure(
        text=f"선택된 항목: {selected_count}개, 총 용량: {total_size / (1024 ** 2):.2f} MB"
    )

# 캐시 항목을 표 형태로 표시하는 기능
def display_cache_items():
    for widget in table_frame.winfo_children():
        widget.destroy()

    # 표 헤더 생성
    headers = ["선택", "Repo ID", "Revision", "Size (MB)", "Last Modified"]
    columns = ["select", "repo_id", "revision", "size", "last_modified"]
    column_widths = [50, 200, 150, 100, 200]  # 각 열의 너비 설정

    for col, (header, col_name, width) in enumerate(zip(headers, columns, column_widths)):
        label = ctk.CTkLabel(master=table_frame, text=header, font=("Arial", 12, "bold"))
        label.grid(row=0, column=col, padx=5, pady=5)
        label.bind("<Button-1>", lambda e, cn=col_name: sort_by_column(cn))
        label.configure(width=width)
        # 헤더 배경색 변경으로 클릭 가능함을 표시
        label.configure(cursor="hand2")

    global checkboxes
    checkboxes = []

    # 데이터 표시
    for row, rev in enumerate(revisions, start=1):
        var = ctk.BooleanVar()
        cb = ctk.CTkCheckBox(master=table_frame, text="check", variable=var, command=update_selection_summary)
        cb.grid(row=row, column=0, padx=5, pady=5)
        checkboxes.append((var, rev))

        # Repo ID
        repo_id_label = ctk.CTkLabel(master=table_frame, text=rev['repo_id'])
        repo_id_label.grid(row=row, column=1, padx=5, pady=5)
        repo_id_label.configure(width=column_widths[1])

        # Revision
        revision_label = ctk.CTkLabel(master=table_frame, text=rev['revision'][:7])
        revision_label.grid(row=row, column=2, padx=5, pady=5)
        revision_label.configure(width=column_widths[2])

        # Size (MB)
        size_label = ctk.CTkLabel(master=table_frame, text=f"{rev['size'] / (1024 ** 2):.2f}")
        size_label.grid(row=row, column=3, padx=5, pady=5)
        size_label.configure(width=column_widths[3])

        # Last Modified
        last_modified_label = ctk.CTkLabel(master=table_frame, text=rev['last_modified'])
        last_modified_label.grid(row=row, column=4, padx=5, pady=5)
        last_modified_label.configure(width=column_widths[4])

    # 전체 선택/해제 버튼
    select_all_btn = ctk.CTkButton(master=table_frame, text="전체 선택", command=select_all)
    select_all_btn.grid(row=len(revisions) + 1, column=0, padx=5, pady=5)

    deselect_all_btn = ctk.CTkButton(master=table_frame, text="전체 해제", command=deselect_all)
    deselect_all_btn.grid(row=len(revisions) + 1, column=1, padx=5, pady=5)

    update_selection_summary()

# 열 정렬 함수
def sort_by_column(column_name):
    global revisions, sort_orders
    reverse = not sort_orders[column_name]
    sort_orders[column_name] = reverse

    if column_name == "size":
        revisions.sort(key=lambda x: x[column_name], reverse=reverse)
    elif column_name == "last_modified":
        revisions.sort(key=lambda x: x[column_name] or '', reverse=reverse)
    else:
        revisions.sort(key=lambda x: x[column_name].lower(), reverse=reverse)

    display_cache_items()

# 전체 선택 기능
def select_all():
    for var, _ in checkboxes:
        var.set(True)
    update_selection_summary()

# 전체 해제 기능
def deselect_all():
    for var, _ in checkboxes:
        var.set(False)
    update_selection_summary()

# 선택한 캐시 항목 삭제
def delete_selected():
    selected_revisions = [rev['revision'] for var, rev in checkboxes if var.get()]
    if not selected_revisions:
        messagebox.showinfo("알림", "삭제할 항목을 선택하세요.")
        return

    confirm = messagebox.askyesno("확인", f"{len(selected_revisions)}개의 수정 버전을 삭제하시겠습니까?")
    if confirm:
        delete_strategy = cache_info.delete_revisions(*selected_revisions)
        delete_strategy.execute()
        messagebox.showinfo("완료", "선택한 캐시가 삭제되었습니다.")
        scan_cache()
    else:
        messagebox.showinfo("취소", "삭제가 취소되었습니다.")

# 로그인 기능
def login():
    token = entry_token.get().strip()
    if token:
        try:
            api.set_access_token(token)
            save_login_token(token)
            label_status.configure(text="로그인 성공!")
        except Exception as e:
            messagebox.showerror("로그인 오류", f"로그인에 실패했습니다: {e}")
    else:
        messagebox.showerror("오류", "유효한 토큰을 입력하세요.")

# 로그아웃 기능
def logout():
    api.set_access_token(None)
    delete_login_token()
    label_status.configure(text="로그아웃되었습니다.")

# 환경 정보 출력
def show_env():
    try:
        env_info = api.whoami()
        label_env_info.configure(text=f"환경 정보: {env_info}")
    except Exception as e:
        label_env_info.configure(text=f"환경 정보를 불러오지 못했습니다: {e}")

# 현재 사용자 정보 출력
def show_whoami():
    try:
        user_info = api.whoami()
        label_user_info.configure(text=f"사용자: {user_info['name']}")
    except Exception as e:
        label_user_info.configure(text=f"사용자 정보를 불러오지 못했습니다: {e}")

# TabView 생성
tab_view = ctk.CTkTabview(app)
tab_view.pack(fill='both', expand=True, padx=20, pady=20)

# 첫 번째 탭: 로그인 / 로그아웃 및 사용자 정보
tab_user = tab_view.add("로그인 및 사용자 정보")

frame_login = ctk.CTkFrame(master=tab_user)
frame_login.pack(pady=10)

label_token = ctk.CTkLabel(master=frame_login, text="Hugging Face 토큰:")
label_token.pack(side="left", padx=10)

entry_token = ctk.CTkEntry(master=frame_login, width=300)
entry_token.pack(side="left")

button_login = ctk.CTkButton(master=frame_login, text="로그인", command=login)
button_login.pack(side="left", padx=10)

button_logout = ctk.CTkButton(master=frame_login, text="로그아웃", command=logout)
button_logout.pack(side="left", padx=10)

label_status = ctk.CTkLabel(master=tab_user, text="")
label_status.pack(pady=10)

# 사용자 정보 및 환경 정보 버튼
button_whoami = ctk.CTkButton(master=tab_user, text="현재 사용자 정보", command=show_whoami)
button_whoami.pack(pady=5)

button_env = ctk.CTkButton(master=tab_user, text="환경 정보 보기", command=show_env)
button_env.pack(pady=5)

# 사용자 정보 및 환경 정보 출력 라벨
label_user_info = ctk.CTkLabel(master=tab_user, text="")
label_user_info.pack(pady=5)

label_env_info = ctk.CTkLabel(master=tab_user, text="")
label_env_info.pack(pady=5)

# 두 번째 탭: 캐시 관리
tab_cache = tab_view.add("캐시 관리")

# 프레임 생성 (스크롤바와 캔버스를 포함할 컨테이너)
container = tk.Frame(tab_cache)
container.pack(fill=tk.BOTH, expand=True)

# 캔버스 생성
canvas = tk.Canvas(container)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 스크롤바 생성
vsb = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
vsb.pack(side=tk.RIGHT, fill=tk.Y)
hsb = tk.Scrollbar(tab_cache, orient="horizontal", command=canvas.xview)
hsb.pack(side=tk.BOTTOM, fill=tk.X)

canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

# 표를 담을 프레임 생성
table_frame = tk.Frame(canvas)

# 캔버스에 프레임 추가
canvas_window = canvas.create_window((0,0), window=table_frame, anchor='nw')

# 스크롤 영역 설정 함수
def configure_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
    # 캔버스 너비를 프레임의 너비로 설정
    canvas.itemconfigure(canvas_window, width=event.width)

# 프레임 크기 변경 시 스크롤 영역 재설정
table_frame.bind('<Configure>', configure_scroll_region)

# 캔버스 크기 변경 시 프레임 너비 조정
canvas.bind('<Configure>', lambda e: canvas.itemconfigure(canvas_window, width=e.width))

# 마우스 휠 스크롤 함수
def on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

# 마우스 휠 이벤트 바인딩
canvas.bind_all("<MouseWheel>", on_mousewheel)

# 표를 담을 프레임 생성
table_frame = tk.Frame(canvas)

canvas.create_window((0,0), window=table_frame, anchor='nw')

# 선택된 항목과 용량 요약 표시 라벨
label_selection_summary = ctk.CTkLabel(master=tab_cache, text="선택된 항목: 0개, 총 용량: 0.00 MB")
label_selection_summary.pack(pady=10)

delete_btn = ctk.CTkButton(master=tab_cache, text="선택한 캐시 삭제", command=delete_selected)
delete_btn.pack(pady=10)

# 프로그램 시작 시 로그인 상태 확인
token = load_login_token()
if token:
    label_status.configure(text="로그인 상태 유지됨")
else:
    label_status.configure(text="로그인 필요")

# 캐시 스캔 실행
scan_cache()

# 메인 루프 시작
app.mainloop()