import customtkinter as ctk
import threading
import predictListen
import sounddevice as sd
import soundfile as sf
import os

classes = ["Dur", "Geri", "Ileri", "Kalk", "Unknown"]

os.makedirs("data", exist_ok=True)

for c in classes:
    os.makedirs(os.path.join("data", c), exist_ok=True)
# def save_last_detection():
#
#     if predictListen.last_audio is None:
#         save_label.configure(text="No audio to save.")
#         return
#
#     label = selected_class.get()
#
#     folder = os.path.join("data", label)
#
#     filename = next_filename(folder, label)
#
#     sf.write(
#         filename,
#         predictListen.last_audio,
#         16000
#     )
#
#     save_label.configure(
#         text=f"Saved:\n{os.path.basename(filename)}"
#     )
#     sf.write(
#         filename,
#         predictListen.last_audio,
#         16000
#     )
#
#     save_label.configure(
#         text=f"Saved:\n{os.path.basename(filename)}"
#     )
#
#     predictListen.last_audio = None


def next_filename(folder, label):

    files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    nums = []

    for f in files:
        try:
            nums.append(int(f.split("_")[-1].split(".")[0]))
        except:
            pass

    n = max(nums)+1 if nums else 1

    return os.path.join(folder, f"{label}_{n:03d}.wav")


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()

root.title(
    "Voice Command Recognition"
)

root.geometry(
    "500x400"
)

status_label = ctk.CTkLabel(
    root,
    text="Listening...",
    font=("Arial",24)
)

status_label.pack(pady=20)

confidence_label = ctk.CTkLabel(
    root,
    text="Confidence: 0%",
    font=("Arial",18)
)

confidence_label.pack()

level_bar = ctk.CTkProgressBar(
    root,
    width=300
)

level_bar.pack(pady=20)

level_bar.set(0)

second_confidence_label = ctk.CTkLabel(
    root,
    text="Second Confidence: 0%",
    font=("Arial",14)
)
second_confidence_label.pack()

end_label = ctk.CTkLabel(
    root,
    text="",
    font=("Arial",14)
)
end_label.pack(pady=5)

# record_title = ctk.CTkLabel(
#     root,
#     text="Dataset Recorder",
#     font=("Arial",18)
# )
# record_title.pack(pady=(20,5))


selected_class = ctk.StringVar(value="Unknown")

# menu = ctk.CTkOptionMenu(
#     root,
#     values=["Dur","Geri","Ileri","Kalk","Unknown"],
#     variable=selected_class
# )
# menu.pack()

# save_label = ctk.CTkLabel(
#     root,
#     text=""
# )
# save_label.pack()

# save_button = ctk.CTkButton(
#     root,
#     text="Save Last Detection",
#     command= save_last_detection
# )
# save_button.pack(pady=10)


def update_gui():

    confidence_label.configure(
        text=f"Confidence: {predictListen.best_prob:.1f}%"
    )
    status_label.configure(
        text=predictListen.prediction_text
    )
    end_label.configure(
        text=predictListen.second_prediction_text
    )
    second_confidence_label.configure(
        text=f"Second Confidence: {predictListen.second_prob:.1f}%"
    )
    level = min(
        predictListen.energy / 0.05,
        1
    )
    level_bar.set(level)


    root.after(
        100,
        update_gui
    )

# Start microphone thread
threading.Thread(
    target=predictListen.start_listening,
    daemon=True
).start()

update_gui()
root.mainloop()
