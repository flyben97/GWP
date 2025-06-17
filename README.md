# 🌍 Global Warming Potential Prediction Platform

A Flask-based web platform that predicts the **Global Warming Potential (GWP)** of chemical compounds using advanced **Graph Neural Networks (GNNs)**. The system supports multiple prediction models for **GWP20**, **GWP100**, and **GWP500** time horizons, and features seamless molecule visualization through **Ketcher**.

---

## 1. Features

* **🔬 GWP Prediction Models**
  Predict GWP values using three pre-trained GNN architectures:

  * **MPNN (Message Passing Neural Network)**
  * **Neural Fingerprint (NF)**
  * **Graph Convolutional Network (GCN)**

* **🧪 SMILES Input Support**
  Input SMILES strings manually, or use the integrated **Ketcher** molecule editor to draw molecules.

* **🌐 Interactive Web Interface**

  * Built with **Flask** and styled using **Tailwind CSS**
  * Responsive UI with model selection and prediction history
  * Integration with **SweetAlert2** for user-friendly alerts

* **🧠 Model Inference**
  Powered by **PyTorch**, **DGL**, and **dgllife** for efficient molecular graph handling and inference.

---

## 2. Prerequisites

Make sure you have the following installed:

* Python 3.8+
* [PyTorch](https://pytorch.org/)
* [DGL (Deep Graph Library)](https://www.dgl.ai/)
* [dgllife](https://lifesci.dgl.ai/)
* Flask
* pandas

> 💡 **Note:** Tailwind CSS and SweetAlert2 are included via CDN—no setup required.

---

## 3. Installation

1. **Clone the Repository**

```bash
git clone https://github.com/flyben97/GWP.git
cd GWP
```

2. **Install Dependencies**

```bash
pip install torch dgl dgllife flask pandas
```

---

## 4. Usage

1. **Start the Flask App**

```bash
python app.py
```

2. **Access the Interface**
   Open your browser and navigate to:
   👉 [http://localhost:5000](http://localhost:5000)

3. **Make Predictions**

   * Select a time horizon model: **GWP20**, **GWP100**, or **GWP500**
   * Enter or draw a molecule (SMILES input)
   * Click **"Predict"** to get the estimated GWP value
   * View prediction history below the input form

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Developed by the **Shanghai Institute of Organic Chemistry**, Chinese Academy of Sciences.
Built with:

* [DGL](https://www.dgl.ai/)
* [dgllife](https://lifesci.dgl.ai/)
* [Ketcher](https://lifescience.opensource.epam.com/ketcher/)



