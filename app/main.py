from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import torch
import efficient_attention_cpp
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    

@app.post("/efficient-attention")
def calculo(seed: int, batch_size: int, seq_len: int, dim: int):
    # Configuración de datos sintéticos
    #seed = 42
    torch.manual_seed(seed)
    #batch_size = 1
    #seq_len = 10
    #dim = 4

    # Generar tensores Q, K, V
    Q = torch.rand((seq_len, dim), dtype=torch.float32)
    K = torch.rand((seq_len, dim), dtype=torch.float32)
    V = torch.rand((seq_len, dim), dtype=torch.float32)

    # Cálculo de atención eficiente en C++
    Q_np, K_np, V_np = Q.numpy(), K.numpy(), V.numpy()
    attention_result = efficient_attention_cpp.efficient_attention(Q_np, K_np, V_np)

    # Convertir resultado a tensor de PyTorch
    attention_result = torch.tensor(attention_result)

    # Mostrar resultados
    #print("Q (Query):\n", Q)
    #print("K (Key):\n", K)
    #print("V (Value):\n", V)
    #print("Efficient Attention Result:\n", attention_result)

    # Visualización
    plt.figure(figsize=(10, 6))
    plt.imshow(attention_result.numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weights")
    plt.title("Efficient Attention Output")
    plt.xlabel("Dimension")
    plt.ylabel("Sequence Position")
    output_file = 'fa_results.png'
    plt.savefig(output_file)
    plt.close()
    # Regresar el archivo como respuesta

    j1 = {
        "Q (Query)": Q,
        "K (Key)": K,
        "V (Value)": V,
        "Efficient Attention Result": attention_result,
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/efficient-attention-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)



    