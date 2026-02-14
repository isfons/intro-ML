# Introducción a redes neuronales

Una introducción práctica a las redes neuronales artificiales utilizando Python y PyTorch, con notebooks interactivos que cubren desde conceptos básicos hasta técnicas avanzadas de optimización.

## Descripción

Este repositorio contiene una serie de Jupyter Notebooks que tratan los siguientes temas:

- **Entrenamiento básico**: Conceptos fundamentales de redes neuronales y paso forward/backward
<a target="_blank" href="https://colab.research.google.com/github/isfons/intro-ML/blob/main/01_nn_training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **Overfitting y Underfitting**: Identificación y prevención del sobreajuste con regularización
<a target="_blank" href="https://colab.research.google.com/github/isfons/intro-ML/blob/main/02_overfit_underfit.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **Ajuste de Hiperparámetros**: Optimización automática de hiperparámetros usando Optuna
<a target="_blank" href="https://colab.research.google.com/github/isfons/intro-ML/blob/main/03_hyperparameter_tuning.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Estructura
```
intro_nn/
├── 01_nn_training.ipynb            # Actividad 1
├── 02_overfit_underfit.ipynb       # Actividad 2
├── 03_hyperparameter_tuning.ipynb  # Actividad 3
├── CCP.csv                         # Dataset de la central de ciclo combinado
├── utils.py                        # Funciones auxiliares para visualización y utilidades
├── pyproject.toml                  # Requisitos para recrear el environment de Python
├── LICENSE                         # Licencia MIT
└── README.md                       # Este archivo
```

## Requisitos e instalación

### Requisitos del sistema
- **Python**: ≥ 3.8
- **GPU** (opcional): CUDA para acelerar entrenamientos

### Instalación
```bash
pip install -e .
```

O instalar manualmente:
```bash
pip install torch>=2.1 \
            numpy>=1.24 \
            pandas>=2.0 \
            matplotlib>=3.7 \
            seaborn>=0.13 \
            scikit-learn>=1.2 \
            tqdm>=4.67 \
            ipykernel \
            ipywidgets \
            ipympl \
            optuna
```

### Librerías principales
| Paquete | Versión | Descripción |
|---------|---------|-------------|
| torch | ≥2.1 | Framework de deep learning |
| scikit-learn | ≥1.2 | Machine learning utilities |
| pandas | ≥2.0 | Manipulación de datos |
| matplotlib | ≥3.7 | Visualización |
| optuna | - | Optimización bayesiana de hiperparámetros |

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

Copyright (c) 2026 Isabela Fons

## 👤 Autor

**Isabela Fons**  
Email: isabela.fons@ua.es

## 📚 Recursos Adicionales

- [Documentación oficial de PyTorch](https://pytorch.org/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
