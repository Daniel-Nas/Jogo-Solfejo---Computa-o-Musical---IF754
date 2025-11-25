import sounddevice as sd
import numpy as np

print("🔍 ESCANEANDO DISPOSITIVOS DE ÁUDIO...\n")

# Pega todos os dispositivos
devices = sd.query_devices()
input_devices = []

print(f"{'ID':<4} | {'NOME DO DISPOSITIVO':<40} | {'CANAIS'}")
print("-" * 60)

for i, dev in enumerate(devices):
    # Filtra só o que tem canal de entrada (Microfones)
    if dev['max_input_channels'] > 0:
        input_devices.append(i)
        print(f"{i:<4} | {dev['name'][:40]:<40} | {dev['max_input_channels']}")

print("-" * 60)
print("\n👉 Olhe a lista acima.")
try:
    escolha = int(input("Digite o NÚMERO (ID) do seu microfone para testar: "))
except:
    print("Número inválido.")
    exit()

print(f"\nTESTANDO DISPOSITIVO ID [{escolha}]...")
print("Fale 'ALÔ' bem alto. (Ctrl+C para parar)")

try:
    def callback(indata, frames, time, status):
        # Calcula volume
        vol = np.linalg.norm(indata) * 10
        barra = "█" * int(vol)
        if vol > 0.5:
            print(f"\r🔊 SOM: {barra[:50]}", end="")
        else:
            print(f"\r...", end="")

    # Abre o microfone escolhido
    with sd.InputStream(device=escolha, channels=1, callback=callback):
        while True:
            sd.sleep(100)
            
except Exception as e:
    print(f"\n❌ ERRO AO ABRIR ESSE MICROFONE:\n{e}")
    print("\nTente outro número da lista.")
except KeyboardInterrupt:
    print("\nFim do teste.")