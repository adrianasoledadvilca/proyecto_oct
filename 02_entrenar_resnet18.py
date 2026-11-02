from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report

transformar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
datos = datasets.ImageFolder("paginas_etiquetadas", transform=transformar)
print("Clases:", datos.classes, "| Páginas:", len(datos))

# Separar por EXPEDIENTE (no por página) para evitar fuga de información
grupos = [Path(ruta).stem.rsplit("_p", 1)[0] for ruta, _ in datos.samples]
etiquetas = [clase for _, clase in datos.samples]
idx_train, idx_test = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    .split(range(len(datos)), etiquetas, grupos)
)
loader_train = DataLoader(Subset(datos, idx_train), batch_size=16, shuffle=True)
loader_test = DataLoader(Subset(datos, idx_test), batch_size=16)

# Transferencia de aprendizaje: se congela ResNet18 y se reentrena solo la última capa
modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for parametro in modelo.parameters():
    parametro.requires_grad = False
modelo.fc = nn.Linear(modelo.fc.in_features, len(datos.classes))

criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.fc.parameters(), lr=1e-3)

for epoca in range(8):
    modelo.train()
    perdida_total = 0.0
    for imagenes, clases in loader_train:
        optimizador.zero_grad()
        perdida = criterio(modelo(imagenes), clases)
        perdida.backward()
        optimizador.step()
        perdida_total += perdida.item()
    print(f"Época {epoca + 1}/8 - pérdida: {perdida_total / len(loader_train):.4f}")

modelo.eval()
reales, predichas = [], []
with torch.no_grad():
    for imagenes, clases in loader_test:
        reales += clases.tolist()
        predichas += modelo(imagenes).argmax(dim=1).tolist()
print(classification_report(
    reales, predichas, labels=list(range(len(datos.classes))),
    target_names=datos.classes, zero_division=0,
))

torch.save({"pesos": modelo.state_dict(), "clases": datos.classes}, "resnet18_estudios.pth")
print("Modelo guardado en 'resnet18_estudios.pth'")