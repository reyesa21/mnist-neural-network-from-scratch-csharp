using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace MNISTNeuralNetworkFromScratch
{
    public partial class MainForm : Form
    {
        // Pixel array to display and modify drawing of digits.
        int[,,] Pixels = new int[60000, 28, 28];

        // Label for each digit, used when updating error in training.
        int[] Label = new int[60000];

        // Label array used with testing set.
        int[] TestLabel = new int[10000];

        int Index = 0;
        int[,] Input = new int[60000, 784];
        int[,] TestInput = new int[10000, 784];

        Random Random = new Random();

        Boolean IsFast = true;
        Boolean IsTesting = false;
        Boolean IsTraining = false;

        const int HlSize = 16;
        double[] InputLayer = new double[784];
        double[] HiddenLayer = new double[HlSize];
        double[] OutputLayer = new double[10];
        double[] OutputDelta = new double[10];
        
        double[,] InputWeights = new double[784, HlSize];
        double[,] HiddenWeights = new double[HlSize, 10];
        double[] HideBiases = new double[HlSize];
        double[] OutputBiases = new double[10];


        int Epoch = 0;
        double Error = 0;
        double[] CorrectOutput = new double[10];
        double LearningRate = 0.3;
        public MainForm()
        {
            InitializeComponent();
        }
        private void MainForm_Load(object sender, EventArgs e)
        {
            // Set input layer and weights and biases.

            for (int j = 0; j < 784; j++)
            {
                InputLayer[j] = (Input[0, j] / 255) * .99 + 0.01;

                for (int k = 0; k < HlSize; k++)
                {
                    InputWeights[j, k] = GetRandomNumber(-1 / Math.Sqrt(784), 1 / Math.Sqrt(784));
                    if (k < 10 && j < HlSize)
                    {
                        HiddenWeights[j, k] = GetRandomNumber(-1 / Math.Sqrt(HlSize), 1 / Math.Sqrt(HlSize));
                    }

                    if (j == 0)
                    {
                        HideBiases[k] = GetRandomNumber(-0.5, 0.5);
                        if (k < 10)
                            OutputBiases[k] = GetRandomNumber(-0.5, 0.5);
                    }
                }
            }

        }

        private void TrainButton_Click(object sender, EventArgs e)
        {
            if (IsTraining == false)
            {
                richTextBox2.Text = "Loading Train Data...";
                richTextBox2.Refresh();
                int z = 0;
                using (var reader = new StreamReader(@"mnist_train.csv"))
                {
                    while (true)
                    {
                        var line = reader.ReadLine();
                        if (line == null)
                            break;
                        var values = line.Split(',');
                        Label[z] = Int32.Parse(values[0]);
                        int k = 0;
                        int j = 0;
                        for (int i = 1; i < values.Length; i++)
                        {
                            Input[z, i - 1] = Int32.Parse(values[i]);
                            Pixels[z, j, k] = Int32.Parse(values[i]);
                            if (i % 28 == 0)
                            {
                                k++;
                                j = 0;
                            }
                            else
                                j++;
                        }
                        z++;
                        if (z == 6000)
                            break;
                    }
                }
                IsTraining = true;
            }
            neuralNetwork(1);
            textBox2.Text = "Epochs Trained: " + Epoch;
            textBox2.Refresh();
            richTextBox1.Text = "";
            for (int i = 0; i < HlSize; i++)
            {
                richTextBox1.Text += "Neuron " + i + "\n Weights: ";
                for (int k = 0; k < 10; k++)
                {
                    richTextBox1.Text += HiddenWeights[i, k] + ", ";
                }
                richTextBox1.Text += "\n Bias: " + HideBiases[i];

                richTextBox1.Text += "\n";
            }


            richTextBox5.Text = "";
            for (int i = 0; i < 10; i++)
            {
                richTextBox5.Text += "Neuron " + i;
                richTextBox5.Text += "\n Bias: " + HideBiases[i];
                richTextBox5.Text += "\n";
            }

        }

        private void updatePicture()
        {
            Color pixel;
            Bitmap flag = new Bitmap(28, 28);
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {

                    if (Pixels[Index, i, j] < 10)
                    {
                        pixel = Color.White;
                        flag.SetPixel(i, j, pixel);

                    }
                    else
                    {
                        pixel = Color.Black;
                        flag.SetPixel(i, j, pixel);
                    }

                }
            }

            Size newSize = new Size(250, 250);
            Image newImg = resizeImage(flag, newSize);
            pictureBox1.Image = newImg;
            if (IsFast == false)
                pictureBox1.Refresh();
            Index++;
        }

        public static Image resizeImage(Image imgToResize, Size size)
        {
            return (Image)(new Bitmap(imgToResize, size));
        }
        public void neuralNetwork(int epochs)
        {
            Epoch++;

            Error = 0;

            //training the neural network
            for (int g = 0; g < epochs; g++)
            {
                int accurate = 0;
                int prevAcc = 0;

                for (int i = 0; i < 6000; i++)
                {

                    if (i % 1000 == 0)
                    {
                        richTextBox2.Text = "Training input: " + i + " - " + (i + 999);
                        richTextBox2.Refresh();
                        richTextBox3.Text = "Training Accuracy: " + (((accurate - prevAcc) / 1000.0) * 100) + "%";
                        richTextBox3.Refresh();
                        prevAcc = accurate;
                    }

                    int rand = Random.Next(0, 60000);
                    rand = i;

                    for (int j = 0; j < 784; j++)
                        InputLayer[j] = (Input[rand, j] / 255.0);

                    for (int k = 0; k < 10; k++)
                        CorrectOutput[k] = 0;

                    CorrectOutput[Label[rand]] = 1;

                    for (int j = 0; j < HlSize; j++)
                    {
                        double vectorMult = 0;

                        for (int k = 0; k < 784; k++)
                            vectorMult += InputWeights[k, j] * InputLayer[k];

                        HiddenLayer[j] = Sigmoid(vectorMult + HideBiases[j]);
                    }

                    for (int j = 0; j < OutputLayer.Length; j++)
                    {

                        double vectorMult = 0;

                        for (int k = 0; k < HlSize; k++)
                            vectorMult += HiddenWeights[k, j] * HiddenLayer[k];

                        OutputLayer[j] = Sigmoid(vectorMult + OutputBiases[j]);
                    }

                    double errorMult = 0;
                    for (int j = 0; j < 10; j++)
                        errorMult += Math.Pow(OutputLayer[j] - CorrectOutput[j], 2);

                    Error += errorMult / 2;

                    for (int j = 0; j < HlSize; j++)
                    {
                        for (int k = 0; k < 10; k++)
                        {
                            if (j == 0)
                            {
                                OutputDelta[k] = (OutputLayer[k] - CorrectOutput[k]) * (OutputLayer[k] * (1 - OutputLayer[k]));
                                OutputBiases[k] -= LearningRate * OutputDelta[k];
                            }
                            HiddenWeights[j, k] -= LearningRate * OutputDelta[k] * HiddenLayer[j];
                        }
                    }

                    for (int j = 0; j < 784; j++)
                        for (int k = 0; k < HlSize; k++)
                        {
                            double hiddenSum = 0;
                            double[] yo = new double[HlSize];
                            for (int l = 0; l < 10; l++)
                                hiddenSum += HiddenWeights[k, l] * OutputDelta[l];

                            yo[k] = LearningRate * (HiddenLayer[k] * (1 - HiddenLayer[k])) * hiddenSum;

                            if (j == 0)
                            {
                                HideBiases[k] -= yo[k];
                            }
                            InputWeights[j, k] -= InputLayer[j] * yo[k];

                        }

                    double maxVal = OutputLayer.Max();
                    int maxIndex = OutputLayer.ToList().IndexOf(maxVal);
                    if (maxIndex == Label[rand])
                        accurate++;


                }
                richTextBox3.Text = "Training Accuracy: " + (((accurate - prevAcc) / 1000.0) * 100) + "%";
                richTextBox3.Refresh();
            }
            textBox1.Text = "Average Error: " + Error / 6000;
            richTextBox2.Text = "Training done!";
            richTextBox2.Refresh();
        }

        public double GetRandomNumber(double minimum, double maximum)
        {

            return Random.NextDouble() * (maximum - minimum) + minimum;
        }

        public static double Sigmoid(double value)
        {
            return (1.0 / (1.0 + Math.Pow(Math.E, -value)));
        }

        private void TestButton_Click(object sender, EventArgs e)
        {
            if (IsTesting == false)
            {
                richTextBox2.Text = "Loading Test Data...";
                richTextBox2.Refresh();

                int z = 0;
                using (var reader = new StreamReader(@"mnist_test.csv"))
                {
                    while (true)
                    {
                        var line = reader.ReadLine();
                        if (line == null)
                            break;
                        var values = line.Split(',');
                        TestLabel[z] = Int32.Parse(values[0]);
                        int k = 0;
                        int j = 0;
                        for (int i = 1; i < values.Length; i++)
                        {
                            TestInput[z, i - 1] = Int32.Parse(values[i]);
                            Pixels[z, j, k] = Int32.Parse(values[i]);
                            if (i % 28 == 0)
                            {
                                k++;
                                j = 0;
                            }
                            else
                                j++;
                        }
                        z++;
                    }
                }

                IsTesting = true;
            }

            int correct = 0;
            for (int u = 0; u < 10000; u++)
            {
                updatePicture();
                for (int j = 0; j < 784; j++)
                {
                    InputLayer[j] = (TestInput[u, j] / 255.0);
                }

                if (u % 1000 == 0)
                {
                    richTextBox2.Text = "Testing input: " + u + " - " + (u + 999);
                    richTextBox2.Refresh();
                    richTextBox6.Text = "Correct Predictions: " + correct + " / " + u;
                    richTextBox6.Refresh();
                }
                double[] correctOutput = new double[10];

                for (int k = 0; k < 10; k++)
                    correctOutput[k] = 0;
                correctOutput[TestLabel[u]] = 1;

                for (int j = 0; j < HlSize; j++)
                {
                    double vectorMult = 0;
                    for (int k = 0; k < 784; k++)
                        vectorMult += InputWeights[k, j] * InputLayer[k];
                    HiddenLayer[j] = Sigmoid(vectorMult + HideBiases[j]);
                    //System.Console.WriteLine(hiddenLayer[j].ToString());
                }

                for (int j = 0; j < OutputLayer.Length; j++)
                {
                    double vectorMult = 0;
                    for (int k = 0; k < HlSize; k++)
                        vectorMult += HiddenWeights[k, j] * HiddenLayer[k];
                    OutputLayer[j] = Sigmoid(vectorMult + OutputBiases[j]);
                    //System.Console.WriteLine("output[ " + j + "]: " + outputLayer[j]);

                }
                double maxVal = OutputLayer.Max();
                int maxIndex = OutputLayer.ToList().IndexOf(maxVal);
                richTextBox4.Text = " " + maxIndex;
                if (maxIndex == TestLabel[u])
                    correct++;
                if (IsFast == false)
                    richTextBox4.Refresh();
            }
            richTextBox2.Text = "Testing done!";
            richTextBox2.Refresh();
            richTextBox6.Text = "Testing Accuracy: " + correct / 10000.0 * 100 + "%";
            richTextBox6.Refresh();
        }

        private void DrawButton_CheckedChanged(object sender, EventArgs e)
        {
            if (drawButton.Checked)
                IsFast = true;
            else
                IsFast = false;
        }
    }
}
