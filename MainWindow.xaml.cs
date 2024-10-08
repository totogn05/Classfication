using System.Windows;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.IO;
using System.Drawing.Imaging;
using System.Windows.Input;

namespace Classfication
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private List<BitmapImage> images = new List<BitmapImage>(); // list to store pictures
        private int currentImageIndex = 0; // trace the index of current image

        private InferenceSession? session;
        private BitmapImage Image = new BitmapImage();
        private Bitmap? originalImage; // store original Image 
        private float currentContrastValue = 0; // store current Contrast Value
        public MainWindow()
        {
            InitializeComponent();
            //設置滑桿的初始值顯示
            ImageScaleValue.Text = $"{ImageScaleSlider.Value:F0}%";
            ContrastValue.Text = $"{ContrastSlider.Value:F0}%";
        }
        private void Select_file_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "ONNX model files (*.onnx)|*.onnx|All files (*.*)|*.*";
            if (openFileDialog.ShowDialog() == true)
            {
                Model_Name_txt.Text = Path.GetFileName(openFileDialog.FileName);
                // import model
                session = new InferenceSession(openFileDialog.FileName);
                foreach (var input in session.InputMetadata)
                {
                    var inputInfo = input.Value;
                    // input number of image 
                    var dimensions = inputInfo.Dimensions;
                    Input_ImageNumber_txt.Text = "輸入張數 : " + dimensions[0].ToString();
                    Input_ImageChannel_txt.Text = "輸入影像通道數 : " + dimensions[1].ToString();
                    Input_ImageSize_txt.Text = "輸入影像尺寸 : " + dimensions[2].ToString() + "X" + dimensions[3].ToString();
                }
                foreach (var Output in session.OutputMetadata)
                {
                    var OutputInfo = Output.Value;
                    Output_ImageSize_txt.Text = OutputInfo.Dimensions[1].ToString() + "種結果";
                }

            }
        }


        private void Vertify_btn(object sender, RoutedEventArgs e)
        {
            // 預處理圖片
            var inputTensor = PreprocessImage(BitmapImageToBitmap(Image), 640, 640);
            // 
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("images", inputTensor) };
            using var results = session?.Run(inputs);

            var output = results?.FirstOrDefault()?.AsTensor<float>();
            if (output != null)
            {
                string[] Result = new string[] { "焊接良好", "燒穿", "焊接中有汙染", "焊接處有縫隙", "缺乏保護氣體", "銲槍移動速度過快" };
                float[] probabilities = output.ToArray();
                int predictedClassIndex = Array.IndexOf(probabilities, probabilities.Max());
                float maxProbability = probabilities[predictedClassIndex];

            }
        }
        public static Bitmap BitmapImageToBitmap(BitmapImage bitmapImage)
        {
            using (MemoryStream memoryStream = new MemoryStream())
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bitmapImage));
                encoder.Save(memoryStream);
                return new Bitmap(memoryStream);
            }
        }
        private static Tensor<float> PreprocessImage(Bitmap bitmap, int width, int height)
        {
            var resizedBitmap = new Bitmap(bitmap, new System.Drawing.Size(width, height));

            float[] imageArray = new float[width * height * 3];

            int index = 0;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = resizedBitmap.GetPixel(x, y);
                    imageArray[index++] = pixel.R / 255.0f; // Red
                    imageArray[index++] = pixel.G / 255.0f; // Green
                    imageArray[index++] = pixel.B / 255.0f; // Blue
                }
            }
            var tensor = new DenseTensor<float>(imageArray, new[] { 1, 3, height, width });
            return tensor;
        }

        private void Select_Image_Click(object sender, RoutedEventArgs e)
        {
            // check if user select picture model
            if (MutiplePic_Model.IsChecked == false && SinglePic_Model.IsChecked == false)
            {
                MessageBox.Show("請選取圖片模式", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
                return; 
            }

            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Image files (*.png;*.jpg)|*.png;*.jpg|All files (*.*)|*.*";

            if (MutiplePic_Model.IsChecked == true)
            {
                openFileDialog.Multiselect = true; // allow to select Multiselect
            }

            if (openFileDialog.ShowDialog() == true)
            {
                
                if (MutiplePic_Model.IsChecked == true)
                {
                    images.Clear(); // clear the pictures had selected 
                    foreach (var fileName in openFileDialog.FileNames)
                    {
                        BitmapImage bitmap = new BitmapImage();
                        bitmap.BeginInit();
                        bitmap.UriSource = new Uri(fileName);
                        bitmap.EndInit();
                        images.Add(bitmap); // add new picture to list

                        // 儲存原始圖像
                        originalImage = BitmapImageToBitmap(bitmap);

                    }
                    currentImageIndex = 0;
                    DisplayCurrentImage(); // show first image
                }
                else if (SinglePic_Model.IsChecked == true)
                {
                    // select single picture
                    images.Clear(); // clear the pictures had selected
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.UriSource = new Uri(openFileDialog.FileName);
                    bitmap.EndInit();
                    Vertify_Image.Source = bitmap; // show image on the Vertify_Image
                    Image = bitmap;
                    SelectPic_txt.Text = Path.GetFileName(openFileDialog.FileName);
                    PicSize_txt.Text = bitmap.Width.ToString("0") + "X" + bitmap.Height.ToString("0");
                }

            }

        }


        private void Model_Select(object sender, RoutedEventArgs e)
        {
            // check if SinglePic_Model and MutiplePic_Model are null
            if (SinglePic_Model != null && MutiplePic_Model != null)
            {
                if (sender == SinglePic_Model && SinglePic_Model.IsChecked == true)
                {
                    MutiplePic_Model.IsChecked = false; // cancel multi mode when change to single mode
                }
                else if (sender == MutiplePic_Model && MutiplePic_Model.IsChecked == true)
                {
                    SinglePic_Model.IsChecked = false; // // cancel single mode when change to multi mode
                }
            }
        }

        private void DisplayCurrentImage()
        {
            if (images.Count > 0 && currentImageIndex >= 0 && currentImageIndex < images.Count)
            {
                Vertify_Image.Source = images[currentImageIndex]; // show the index of current image
                SelectPic_txt.Text = Path.GetFileName(images[currentImageIndex].UriSource.LocalPath);
                PicSize_txt.Text = images[currentImageIndex].Width.ToString("0") + "X" + images[currentImageIndex].Height.ToString("0");
            }
        }

        private void Previous_Image_Click(object sender, RoutedEventArgs e)
        {
            if (images.Count > 0)
            {
                currentImageIndex = (currentImageIndex - 1 + images.Count) % images.Count;
                DisplayCurrentImage();
            }
        }

        private void Next_Image_Click(object sender, RoutedEventArgs e)
        {
            if (images.Count > 0)
            {
                currentImageIndex = (currentImageIndex + 1) % images.Count;
                DisplayCurrentImage();
            }
        }

        // 處理滑鼠滾輪改變對比度
        private void ContrastSlider_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (ContrastSlider != null)
            {
                // 如果滑輪向上滾動，對比度加 1，向下滾動，對比度減 1
                if (e.Delta > 0)
                {
                    ContrastSlider.Value += 1;
                }
                else
                {
                    ContrastSlider.Value -= 1;
                }
            }
        }
        private void LoadImage(string path)
        {
            // 在加載新圖像之前，釋放原有的圖像資源
            if (originalImage != null)
            {
                originalImage.Dispose();
                originalImage = null;
            }

            originalImage = new Bitmap(path);
            Vertify_Image.Source = BitmapToBitmapImage(originalImage);
        }
        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (ContrastValue != null)
            {
                ContrastValue.Text = $"{e.NewValue:F0}%";
            }

            // 確保有原始圖像存在
            if (originalImage != null)
            {
                // 檢查是否真的有變化再執行調整，避免每次都重複處理
                float contrastValue = (float)e.NewValue;
                if (contrastValue != currentContrastValue)
                {
                    currentContrastValue = contrastValue;

                    // 調整圖片的對比度，只在對比度變更時才進行
                    // 這裡的 contrastedBitmap 會被重複利用
                    if (Vertify_Image.Source is BitmapImage bitmapImage)
                    {
                        Bitmap contrastedBitmap = AdjustContrast(originalImage, contrastValue);

                        // 儲存調整後的圖片，避免反覆創建
                        Vertify_Image.Source = BitmapToBitmapImage(contrastedBitmap);
                    }
                }
            }
        }

        private Bitmap AdjustContrast(Bitmap image, float contrastValue)
        {
            float contrast = contrastValue / 100.0f;
            float adjustedContrast = 1 + contrast;
            float brightnessOffset = 128 * (1 - adjustedContrast);

            Bitmap newImage = new Bitmap(image.Width, image.Height);
            using (Graphics g = Graphics.FromImage(newImage))
            {
                float[][] matrixElements = {
            new float[] { adjustedContrast, 0, 0, 0, brightnessOffset },
            new float[] { 0, adjustedContrast, 0, 0, brightnessOffset },
            new float[] { 0, 0, adjustedContrast, 0, brightnessOffset },
            new float[] { 0, 0, 0, 1, 0 },
            new float[] { 0, 0, 0, 0, 1 }
        };

                ColorMatrix colorMatrix = new ColorMatrix(matrixElements);

                using (ImageAttributes attributes = new ImageAttributes())
                {
                    attributes.SetColorMatrix(colorMatrix);
                    g.DrawImage(image, new Rectangle(0, 0, image.Width, image.Height), 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, attributes);
                }
            }

            return newImage; // 返回新的對比度調整後的圖像
        }

        private BitmapImage BitmapToBitmapImage(Bitmap bitmap)
        {
            using (MemoryStream memoryStream = new MemoryStream())
            {
                bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Png);
                memoryStream.Position = 0;
                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memoryStream;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze(); // freeze BitmapImage，avoid extra Memory occupied
                return bitmapImage;
            }
        }
        private void ResetContrast()
        {
            // 重置滑桿和圖像
            ContrastSlider.Value = 0;
            Vertify_Image.Source = BitmapToBitmapImage(originalImage);
            currentContrastValue = 0;
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            ResetContrast(); // 當按下重置按鈕時，呼叫重置方法 
        }


        // 縮放圖片的方法
        private void UpdateImageScale(double value)
        {
            if (ImageScaleValue != null)
            {
                ImageScaleValue.Text = $"{value:F0}%";

                // 將滑桿的值從 0 到 100 映射到 1 到 2 的範圍
                double scaleValue = 1 + (value / 100.0); // 0% -> 1, 100% -> 2

                // 更新圖片縮放比例
                ImageScaleTransform.ScaleX = scaleValue;
                ImageScaleTransform.ScaleY = scaleValue;
                if (Vertify_Image.Source != null)
                {
                    // 更新圖片的寬度和高度
                    Vertify_Image.Width = Vertify_Image.Source.Width * scaleValue;
                    Vertify_Image.Height = Vertify_Image.Source.Height * scaleValue;
                }
            }
        }

        private void ResetContrast1()
        {
            // 重置滑桿的值為0
            ContrastSlider.Value = 0;

            // 更新圖片縮放為0%
            UpdateImageScale(0);

            // 重置縮放滑桿的值為0
            ImageScaleSlider.Value = 0;
            UpdateImageScale(0);
        }

        private void ResetButton_Click1(object sender, RoutedEventArgs e)
        {
            ResetContrast1();
        }

        // 滑桿數值變更事件
        private void Slider_ValueChanged_1(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            UpdateImageScale(e.NewValue);
        }

        // ImageScaleSlider 的 MouseWheel 控制事件
        private void ImageScaleSlider_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (e.Delta > 0)
            {
                ImageScaleSlider.Value += 1; // 滾輪向上
            }
            else if (e.Delta < 0)
            {
                ImageScaleSlider.Value -= 1; // 滾輪向下
            }
            UpdateImageScale(ImageScaleSlider.Value);
        }
    }
}