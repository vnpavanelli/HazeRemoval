#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkBilateralImageFilter.h"
#include "QuickView.h"
#define ARMA_NO_DEBUG
#define ARMA_MAT_PREALLOC 3
#define ARMA_USE_BLAS
#define ARMA_USE_SUPERLU
#define ARMA_USE_CXX11
#define ARMA_USE_OPENMP
#include <armadillo>
#include <armadillo_bits/arma_forward.hpp>
#include <stdlib.h>
#include <itkNeighborhood.h>
#include <itkNeighborhoodIterator.h>
#include <cmath>
#include <limits>
#include <math.h>
#include <tuple>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <future>

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
 
  typedef float PixelComponent;
  typedef itk::RGBPixel< PixelComponent >    PixelType;
  typedef itk::RGBPixel< unsigned char >    RGBPixelType;
  typedef itk::Image<PixelType, 2> ImageType;
  typedef itk::Image<PixelComponent, 2> ImageGrayType;
  typedef itk::Image<RGBPixelType, 2> BMPType;
  typedef itk::Image<unsigned char, 2> BMPGrayType;

  typedef itk::BilateralImageFilter<
      ImageGrayType, ImageGrayType >
          FilterType;

  double epsilon = 1e-4; //1e-3;
  double epsilonG = 1e-4;
  double lambda = 1e-4;
//  const unsigned int wk = 9;
  const double wk = 9.0;

  const bool DEBUG = false;
  const bool VERBOSE = false;
  const unsigned int THREADS = 8;
  int largura = 0, altura = 0;
  ImageGrayType::SizeType size;
  std::atomic_int posts;

    std::map<unsigned int, std::pair<unsigned int, unsigned int>> mapa;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> mapa_inv;

    struct Cache {
        arma::fmat cov_inv;
        arma::fmat mean;

        Cache() : mean(3,1), cov_inv(3,3) {};
    };


template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);

template<typename TImageType>
static void WriteFile(std::string filename, typename TImageType::Pointer image);

void minima(typename ImageType::Pointer image_in, typename ImageGrayType::Pointer image_min);
void dark_channel(typename ImageGrayType::Pointer image_min, typename ImageGrayType::Pointer image_dark);
std::vector<double> achaA(typename ImageGrayType::Pointer image_dark, typename ImageType::Pointer image_in);
void tiraHaze(ImageType::Pointer image_in, ImageGrayType::Pointer image_gray, ImageType::Pointer image_out, std::vector<double> pixel_A);
void corrigeA(ImageType::Pointer image_in, ImageType::Pointer image_ac, std::vector<double> A);
void matting(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t);
void matting2(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t);
void guided_filtering(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t);
double laplacian(ImageType::Pointer image_in, unsigned int ix, unsigned int iy, unsigned int jx, unsigned int jy);
double Lij(ImageType::Pointer image_in, int ix, int iy, int jx, int jy, std::map<unsigned int, arma::mat> const&, std::map<unsigned int, arma::mat> const&);
void calculaT(typename ImageGrayType::Pointer image_dark, typename ImageGrayType::Pointer image_t);
arma::mat achaW (ImageType::Pointer image_in, int x, int y);
bool checaDistancia (int, int, int, int, unsigned int distancia=3);
bool checaDistancia (int, int, unsigned int distancia=3);


/*
double calculaL (ImageType::Pointer image, int i, int j, int w, std::vector<Cache> const &cache);
std::vector<int> achaJanelas (int i, int j);
double matting_L (ImageType::Pointer image, int i, int j, std::vector<Cache> const &cache);
arma::mat carregaJanela (ImageType::Pointer image, int w);
void thread_L(ImageType::Pointer image, int i, int total_pixels, arma::sp_mat &L, std::mutex &mtx, std::vector<Cache> const &cache);
*/

inline int ConverteXY2I (int, int);
inline std::pair<int, int> ConverteI2XY (int);
inline ImageType::IndexType ConverteI2Index (int);


void uchar2float(typename ImageType::Pointer image);
void float2uchar(ImageType::Pointer, BMPType::Pointer);
void float2uchar(ImageGrayType::Pointer, BMPGrayType::Pointer);


/* Cabecalhos novos */
void thread_L(ImageType::Pointer image, int ix, int iy, int total_pixels, arma::sp_fmat &L, std::mutex &mtx, std::vector<Cache> const &cache);
double calculaL (ImageType::Pointer image, int ix, int iy, int jx, int jy, std::pair<int, int> wp, std::vector<Cache> const &cache);
arma::fmat carregaJanela (ImageType::Pointer image, int x, int y);
std::vector<std::pair<int, int>> achaJanelas (int ix, int iy, int jx, int jy);
double matting_L (ImageType::Pointer image, int ix, int iy, int jx, int jy, std::vector<Cache> const &cache);

/* Guided Filtering */

arma::fmat fmean(arma::fmat const &I);
arma::fmat fmean2(arma::fmat const &I);
arma::fmat fetchWindow(arma::fmat const &I, int x, int y, int r=3);
float fetchMean(arma::fmat const &I, int x, int y, int r);

int ConverteXY2I (int x, int y) {
    return (y*largura + x);
}

std::pair<int, int> ConverteI2XY (int i) {
    std::pair<int,int> par;
    par.second = i/largura;
    par.first = i % largura;
    return par;
}

ImageType::IndexType ConverteI2Index (int i) {
    int x,y;
    y = i/largura;
    x = i % largura;
    return {x,y};
}
 
int main(int argc, char *argv[])
{
  if(argc < 2)
    {
    std::cerr << "Required: filename" << std::endl;
 
    return EXIT_FAILURE;
    }
  if (argc > 2) {
      lambda = atof(argv[2]);
      epsilon = atof(argv[3]);
      epsilonG = atof(argv[4]);
  }
  std::string inputFilename = argv[1];
 

  ImageType::Pointer image = ImageType::New();
//  ImageType::Pointer image_in = ImageType::New();
  ImageType::Pointer image_out = ImageType::New();
  ImageType::Pointer image_out2 = ImageType::New();
  ImageGrayType::Pointer image_tchapeu = ImageGrayType::New();
  ImageGrayType::Pointer image_t = ImageGrayType::New();
  ImageGrayType::Pointer image_t2 = ImageGrayType::New();
  BMPType::Pointer image_bmp = BMPType::New();
  BMPGrayType::Pointer image_gray_bmp = BMPGrayType::New();


  ReadFile<ImageType>(inputFilename, image);
  
  ImageGrayType::RegionType region;
  ImageGrayType::IndexType start;
//  ImageGrayType::SizeType size;
  start[0] = start[1] = 0;
  size = image->GetLargestPossibleRegion().GetSize();

    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            mapa[j*size[0]+i] = {i,j};
            mapa_inv[{i,j}] = j*size[0]+i;
        }
    }


  std::cout << "Image dimensions: " << size[0] << " x " << size[1] << std::endl;
  largura = size[0]; 
  altura = size[1];

  region.SetSize(size);
  region.SetIndex(start);



  /*
  image_in->SetRegions(region);
  image_in->Allocate();
  */

  image_out->SetRegions(region);
  image_out->Allocate();

  /*
  image_out2->SetRegions(region);
  image_out2->Allocate();
  */

  image_tchapeu->SetRegions(region);
  image_tchapeu->Allocate();

  image_t->SetRegions(region);
  image_t->Allocate();

  image_t2->SetRegions(region);
  image_t2->Allocate();


  std::vector<double> pixel_A;
  uchar2float (image);
  {
  ImageGrayType::Pointer image_min = ImageGrayType::New();
  image_min->SetRegions(region);
  image_min->Allocate();
  minima (image, image_min);

  ImageGrayType::Pointer image_dark = ImageGrayType::New();
  image_dark->SetRegions(region);
  image_dark->Allocate();

  dark_channel (image_min, image_dark);
  pixel_A = achaA(image_dark, image);
  std::cout << "Valor de A atmosferico: " << pixel_A[0] << "," << pixel_A[1] << "," << pixel_A[2] << std::endl;

  ImageType::Pointer image_ac = ImageType::New();
  image_ac->SetRegions(region);
  image_ac->Allocate();
  corrigeA (image, image_ac, pixel_A);
  minima (image_ac, image_min);
  dark_channel (image_min, image_dark);
  calculaT (image_dark, image_tchapeu);

  {
  image_bmp->SetRegions(region);
  image_bmp->Allocate();

  image_gray_bmp->SetRegions(region);
  image_gray_bmp->Allocate();

  float2uchar(image_tchapeu, image_gray_bmp);
  WriteFile<BMPGrayType>(inputFilename + ".tchapeu.bmp", image_gray_bmp);
  float2uchar(image_dark, image_gray_bmp);
  WriteFile<BMPGrayType>(inputFilename + ".dark.bmp", image_gray_bmp);
  float2uchar(image_min, image_gray_bmp);
  WriteFile<BMPGrayType>(inputFilename + ".min.bmp", image_gray_bmp);
  }



  }
  guided_filtering(image, image_tchapeu, image_t);
//  matting2 (image, image_tchapeu, image_t2);

  FilterType::Pointer bilateralFilter = FilterType::New();
  bilateralFilter->SetInput( image_t.GetPointer() );
  bilateralFilter->SetDomainSigma(2.0);
  bilateralFilter->SetRangeSigma(2.0);
  bilateralFilter->Update();
  image_t2 = bilateralFilter->GetOutput();

  //tiraHaze(image, image_t, image_out, pixel_A);
  tiraHaze(image, image_t, image_out, pixel_A);

  std::cout << "Salvando imagens" << std::endl;

  {
  image_bmp->SetRegions(region);
  image_bmp->Allocate();

  image_gray_bmp->SetRegions(region);
  image_gray_bmp->Allocate();


  float2uchar(image_t2, image_gray_bmp);
  WriteFile<BMPGrayType>(inputFilename + ".t2.bmp", image_gray_bmp);
  float2uchar(image_out, image_bmp);
  WriteFile<BMPType>(inputFilename + ".out.bmp", image_bmp);
  }

  std::cout << "Exibindo imagens" << std::endl;

  QuickView viewer;
  viewer.AddImage(image.GetPointer(), true, "in");
//  viewer.AddImage(image_min.GetPointer(), true, "min");
//  viewer.AddImage(image_dark.GetPointer(), true, "dark");
  viewer.AddImage(image_t.GetPointer(), true, "t");
  viewer.AddImage(image_t2.GetPointer(), true, "t2");
  viewer.AddImage(image_tchapeu.GetPointer(), true, "tchapeu");
  viewer.AddImage(image_out.GetPointer(), true, "out");
//  viewer.AddImage(image_out2.GetPointer(), true, "out2");
  viewer.Visualize();

  return EXIT_SUCCESS;
}

template<typename TImageType>
void WriteFile(std::string filename, typename TImageType::Pointer image)
{
    std::cout << "Salvando imagem em " << filename << std::endl;
    typedef  itk::ImageFileWriter< TImageType  > WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
}


 
template<typename TImageType>
void ReadFile(std::string filename, typename TImageType::Pointer image)
{

  typedef itk::ImageFileReader<TImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
 
  reader->SetFileName(filename);
  reader->Update();

  image->Graft(reader->GetOutput());
}

void float2uchar(ImageGrayType::Pointer image, BMPGrayType::Pointer image2) {
  /* transforma imagem unsigned char em float */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image->GetPixel({i,j});
          pixel = std::min(255.0, std::max(0.0, (double) pixel*255));
          image2->SetPixel({i, j}, pixel);
      }
  }
}



void float2uchar(ImageType::Pointer image, BMPType::Pointer image2) {
  /* transforma imagem unsigned char em float */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image->GetPixel({i,j});
          pixel[0] = std::min(255.0, std::max(0.0, (double) pixel[0]*255));
          pixel[1] = std::min(255.0, std::max(0.0, (double) pixel[1]*255));
          pixel[2] = std::min(255.0, std::max(0.0, (double) pixel[2]*255));
          image2->SetPixel({i, j}, pixel);
      }
  }
}



void uchar2float(typename ImageType::Pointer image) {
  /* transforma imagem unsigned char em float */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image->GetPixel({i,j});
          pixel[0] /= 255.0;
          pixel[1] /= 255.0;
          pixel[2] /= 255.0;
          image->SetPixel({i, j}, pixel);
      }
  }
}

void minima(typename ImageType::Pointer image_in, typename ImageGrayType::Pointer image_min) {
  std::cout << "Criando imagem minima..." << std::endl;
  /* Criar imagem com minimo dentre os canais RGB */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image_in->GetPixel({i,j});
          double min = std::min(pixel[0], std::min(pixel[1], pixel[2]));
          image_min->SetPixel({i, j}, min);
      }
  }
}


void calculaT(typename ImageGrayType::Pointer image_dark, typename ImageGrayType::Pointer image_t) {
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          image_t->SetPixel({i,j}, 1.0 - 0.95*image_dark->GetPixel({i,j}));
      }
  }

}

void dark_channel(typename ImageGrayType::Pointer image_min, typename ImageGrayType::Pointer image_dark) {
  const int Patch_window = 7;  // floor(15/2);

    arma::fmat  I(size[0], size[1]);
    std::cout << "Loading image to matrix: " << std::endl;
    for (int i=0; i < size[0]; i++) {
        for (int j=0; j < size[1]; j++) {
            I(i,j) = image_min->GetPixel({i,j});
        }
    }
  std::cout << "Criando Dark Channel..." << std::endl;
  I.each_row([Patch_window](arma::Row<float> &a) {
          arma::Row<float> tmp(a.n_elem);
          for (int i=0; i<a.n_elem;i++) {
            int b = std::max(0,i-Patch_window);
            int e = std::min((int) a.n_elem-1, i+Patch_window-1);
            tmp(i) = a(arma::span(b,e)).min();
          }
          a = tmp;
          });
  I.each_col([Patch_window](arma::Col<float> &a) {
          arma::Row<float> tmp(a.n_elem);
          for (int i=0; i<a.n_elem;i++) {
            int b = std::max(0,i-Patch_window);
            int e = std::min((int) a.n_elem-1, i+Patch_window-1);
            tmp(i) = a(arma::span(b,e)).min();
          }
          a = tmp;
          });

  /* Criar o Prior Dark Channel usando o minimo de um filtro Patch_window X Patch_window (e.g. 15x15) */
  for (int i = 0; i < size[0]; i++) {
      for (int j = 0; j < size[1]; j++) {

          /*
          int i1, i2, j1, j2;
          i1 = std::max(i-Patch_window,0);
          i2 = std::min(i+Patch_window, (int) size[0]-1);
          j1 = std::max(j-Patch_window,0);
          j2 = std::min(j+Patch_window, (int) size[1]-1);
          float min = I(arma::span(i1,i2),arma::span(j1,j2)).min();
          */
          /*
          auto min = image_min->GetPixel({i1,j1});
//          std::cout << "Dark channel para (" << i << "," << j << ") nos intervalos: (" << i1 << "," << j1 << ") e (" << i2 << "," << j2 << ")" << std::endl;
          for (;i1<=i2;i1++) {
//              std::cout << " i1 = " << i1 << "  i2 = " << i2 << std::endl;
              for (int ji = j1; ji<=j2;ji++) {
                  min = std::min(min, image_min->GetPixel({i1, ji}));
//                  std::cout << "  -> min = " << min << "  pixel = " << image_min->GetPixel({i1, ji}) << "  pos = (" << i1 << "," << ji << ")" << std::endl;
              }
          }
          */
          image_dark->SetPixel({i,j}, I(i,j));
//          exit(1);
      }
  }
}


std::vector<double> achaA(typename ImageGrayType::Pointer image_dark, typename ImageType::Pointer image_in) {
  std::cout << "Procurando valor de A..." << std::endl;
  unsigned int total_pixels = size[0]*size[1];
  int porcento01 = ceil((double) total_pixels * 0.001);
  float pixel_max = 1.0f;
  std::cout << "Total pixels: " << total_pixels << std::endl;
  std::cout << "0.1%: " << porcento01 << std::endl;
  std::vector<double> pixel_A({0, 0, 0});

  std::vector<std::pair<double, unsigned int>> pixels;
  pixels.reserve(total_pixels);
  int c = 0;
  PixelComponent *ptr = image_dark->GetBufferPointer();
  while (c < total_pixels) {
      pixels[c] = { *(ptr+c), c };
      c++;
  }
  std::sort(pixels.begin(), pixels.end(), [] (std::pair<double, unsigned int> const &a, std::pair<double, unsigned int> const &b) { return a.first < b.first; });
  PixelType *ptr2 = image_in->GetBufferPointer();
  for (int i = 0; i < porcento01; i++) {
      PixelType *pixel = ptr2+pixels[i].second;
      if (VERBOSE) {
          std::cout << " -> pixel: " << pixels[i].first << " (" << i << "/" << porcento01 << ")" << std::endl;
          std::cout << "    -> ( " << (*pixel)[0] << " , " << (*pixel)[1] << " , " << (*pixel)[2] << " )" << std::endl;
      }
                  pixel_A[0] = std::max(pixel_A[0], (double) (*pixel)[0]);
                  pixel_A[1] = std::max(pixel_A[1], (double) (*pixel)[1]);
                  pixel_A[2] = std::max(pixel_A[2], (double) (*pixel)[2]);

  }

  /*
  do {
      for (unsigned int i = 0; i < size[0] && porcento01 > 0; i++) {
          for (unsigned int j = 0; j < size[1] && porcento01 > 0; j++) {
              auto pixel = image_dark->GetPixel({i,j});
              if (pixel >= pixel_max) {
                  if (VERBOSE) {
                      std::cout << " -> pixel: " << pixel << " >= " << pixel_max << "   porcento01:" << porcento01 << std::endl;
                  }
                  porcento01--;
                  auto pixel = image_in->GetPixel({i,j});
                  pixel_A[0] = std::max(pixel_A[0], (double) pixel[0]);
                  pixel_A[1] = std::max(pixel_A[1], (double) pixel[1]);
                  pixel_A[2] = std::max(pixel_A[2], (double) pixel[2]);
              }
          }
      }
      pixel_max-=.01;
  } while (porcento01>0 && pixel_max >0);
  */
  return pixel_A;
}


void tiraHaze(ImageType::Pointer image_in, ImageGrayType::Pointer image_dark, ImageType::Pointer image_out, std::vector<double> A) {
  const PixelComponent t_max = 0.1;
  std::vector<float> pixel_minimo({1, 1, 1}), 
      pixel_maximo({0, 0, 0});
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          PixelType pixel = image_in->GetPixel({i,j});
          PixelType pixelout;
          //PixelComponent t_dark = 1.0-0.95*(image_dark->GetPixel({i,j}));
          PixelComponent t_dark = image_dark->GetPixel({i,j});
          double t = std::max(t_max,t_dark);

          pixelout[0] = ((double) pixel[0] - A[0])/t + A[0];
          pixelout[1] = ((double) pixel[1] - A[1])/t + A[1];
          pixelout[2] = ((double) pixel[2] - A[2])/t + A[2];

          pixel_minimo[0] = std::min(pixel_minimo[0], pixelout[0]);
          pixel_minimo[1] = std::min(pixel_minimo[1], pixelout[1]);
          pixel_minimo[2] = std::min(pixel_minimo[2], pixelout[2]);

          pixel_maximo[0] = std::max(pixel_maximo[0], pixelout[0]);
          pixel_maximo[1] = std::max(pixel_maximo[1], pixelout[1]);
          pixel_maximo[2] = std::max(pixel_maximo[2], pixelout[2]);

          pixelout[0] = std::min(1.0, std::max(0.0, (double) pixelout[0]));
          pixelout[1] = std::min(1.0, std::max(0.0, (double) pixelout[1]));
          pixelout[2] = std::min(1.0, std::max(0.0, (double) pixelout[2]));

          if (DEBUG || VERBOSE) {
          std::cout << " -> (" << i << "," << j << ") t: " << (double) t << " tx: " << t_dark <<  " pixel0: " << (double) pixel[0] << " pixelout0: " << (double) pixelout[0] 
                    << " pixel1: " << (double) pixel[1] << " pixelout1: " << (double) pixelout[1]
                    << " pixel2: " << (double) pixel[2] << " pixelout2: " << (double) pixelout[2]
//               << " P-A:" << (double) pixel[0]-A[0] << " P-A/t: " <<(double)  (pixel[0]-A[0])/t
              << std::endl;
          }
          image_out->SetPixel({i,j}, pixelout);
      }
  }
  std::cout << "TiraHaze!!!!" << std::endl
            << "Pixel minimo: ( " << pixel_minimo[0] << " , " << pixel_minimo[1] << " , " << pixel_minimo[2] << " ) " << std::endl
            << "Pixel maximo: ( " << pixel_maximo[0] << " , " << pixel_maximo[1] << " , " << pixel_maximo[2] << " ) " << std::endl;
}

void corrigeA(ImageType::Pointer image_in, ImageType::Pointer image_ac, std::vector<double> A) {
  if (VERBOSE) std::cout << "corrigeA Iniciando" << std::endl;

  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image_in->GetPixel({i,j});
          std::vector<double> pixel2;
          pixel2.resize(3);
          if (VERBOSE) {
              std::cout << " Pixel (" << i << "," << j << "): ( " << pixel[0] << " , " << pixel[1] << " , " << pixel[2]
                  << " ) => ( " << pixel[0]/A[0] << " , " << pixel[1]/A[1] << " , " << pixel[2]/A[2] << " ) " << std::endl;
          }
          pixel2[0] = pixel[0] / A[0];
          pixel2[1] = pixel[1] / A[1];
          pixel2[2] = pixel[2] / A[2];
          if (VERBOSE) if (pixel[0] >= 1 || pixel[1] >= 1 || pixel[2] >= 1) std::cout << "*********************************************" << std::endl;
          pixel[0] = std::min(1.0, std::max(0.0, pixel2[0]));
          pixel[1] = std::min(1.0, std::max(0.0, pixel2[1]));
          pixel[2] = std::min(1.0, std::max(0.0, pixel2[2]));
          image_ac->SetPixel({i,j}, pixel);
      }
  }
}

void checaL (arma::sp_mat &L) {
    using namespace std;
    cout << "Checando matriz L" << endl;
    cout << "Checando simetria..." << endl;
    for (int i = 0; i < L.n_rows; i++) {
        for (int j = i+1; j < L.n_cols; j++) {
            if (i != j) {
                if (L(i,j) != L(j,i)) {
                    cout << "Erro na simetria!!! (" << i << "," << j << ") -> " << L(i,j) << "!=" << L(j,i) << endl;
                }
            }
        }
    }
    for (int i = 0; i < L.n_rows; i++) {
        double soma=0.0;
        for (int j=0; j < L.n_cols; j++) {
            if (i!=j) {
                soma += L(i,j);
            }
        }
        soma *= -1.0;
        if (abs(L(i,i)-soma) > 1e-6) cout << "L(" << i << "," << i << ") = " << L(i,i) << " != " << soma << endl;
        L(i,i) = soma;
    }
}

bool checaDistancia (int ix, int iy, int jx, int jy, unsigned int distancia) {
    if ( abs(ix-jx) < distancia && abs(iy-jy) < distancia) {
        return true;
    }
    return false;
}

bool checaDistancia (int i, int j, unsigned int distancia) {
    return checaDistancia(mapa[i].first, mapa[i].second, mapa[j].first, mapa[j].second, distancia);
}

void matting(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t) {
    unsigned int total_pixels = size[0]*size[1];
    std::cout << "Total pixels: " << total_pixels << std::endl;

    /*
    std::map<unsigned int, std::pair<unsigned int, unsigned int>> mapa;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> mapa_inv;
    */
    std::map<unsigned int, arma::mat> Wcov, Wmean;

    std::cout << "Fazendo Wcov e Wmean..." << std::endl;
    for (int i=0; i<size[0]; i++) {
        for (int j=0; j<size[1]; j++) {
            auto W = achaW(image_in, i, j);
//            Wcov[mapa_inv[{i,j}]] = arma::cov(W);
            unsigned int pos = mapa_inv[{i,j}];
            Wmean[pos] = arma::mean(W).t();
            Wcov[pos] = W.t()*W/9 - Wmean[pos]*Wmean[pos].t();
            /*
            W.print(std::cout, "W");
            arma::cov(W).print(std::cout, "Wcov");
            arma::mean(W).print(std::cout, "Wmean");
            exit(1);
            */
        }
    }

    arma::sp_mat L(total_pixels, total_pixels);

    std::cout << "Preenchendo L: " << std::flush;
    for (int i = 0; i < total_pixels; i++) {
        if (i % 100 == 0) std::cout << " -> " << i << " de " << total_pixels << std::endl;
        double soma = 0.0;
        for (int j=0; j < i; j++) {
            soma += L(i,j);
        }
        for (int y=mapa[i].second; y<std::min(mapa[i].second+3,(unsigned int) size[1]); y++) {
            for (int x=mapa[i].first; x<std::min(mapa[i].first+3,(unsigned int) size[0]); x++) {
//                std::cout << " Lij para (" << x << "," << y << ")" << std::endl;
                if (checaDistancia(mapa[i].first, mapa[i].second, x, y)) {
//                    std::cout << " Distancia ok" << std::endl;
                    int j = mapa_inv[{x,y}];
//                    std::cout << "   i = " << i << "  j = " << j << std::endl;
                    if (i != j) {
                        double tmp = Lij(image_in, mapa[i].first, mapa[i].second, x, y, Wcov, Wmean);
                        L(i,j) = L(j,i) = tmp;
                        soma += tmp;
                    }
                }
            }
        }
        L(i,i) = -1.0 * soma + lambda;
    }

    std::cout << "L preenchida" << std::endl;


    std::cout << "salvando em txt" << std::endl;
    if (DEBUG) {
        std::fstream fs;
        fs.open("L.txt", std::fstream::out);
        for (int i = 0; i < L.n_rows ; i++) {
            for (int j = 0; j < L.n_cols; j++) {
                fs << L(i,j) << " ";
            }
            fs << "\n";
        }
        fs.close();
    }
    /*
    L.print(std::cout);
    checaL(L);
    exit(0);
    */
    std::cout << "Fazendo L + lambda" << std::endl;

    //L += arma::eye(total_pixels, total_pixels)*lambda;
    //L += arma::eye(total_pixels, total_pixels)*1e3;

    std::cout << "Fazendo tchapeu" << std::endl;
    arma::mat tchapeu(total_pixels,1);
    for (int i = 0; i < total_pixels; i++) {
        tchapeu(i,0) = image_tchapeu->GetPixel({mapa.at(i).first, mapa.at(i).second}) * lambda;
    }
    /*
    unsigned int c=0;
    auto *ptr = image_tchapeu->GetBufferPointer();
    while (c < total_pixels) {
        tchapeu(c,0) = *(ptr+c) * lambda;
        c++;
    }
    */

    if (DEBUG || VERBOSE) {
        std::cout << "Salvando tchapeu em TXT" << std::endl;
        std::fstream fs;
        fs.open("tchapeu.txt", std::fstream::out);
        for (int i=0; i<total_pixels; i++) {
            fs << tchapeu(i,0) << "\n";
        }
        fs.close();
    }

    std::cout << "tchapeu: " << tchapeu(0,0) << " " << tchapeu(1,0) << " " << tchapeu(2,0) << std::endl;

    std::cout << "Fazendo spsolve" << std::endl;
    auto t = arma::spsolve(L,tchapeu);

    std::cout << "t: " << t(0,0) << " " << t(1,0) << " " << t(2,0) << std::endl;

    std::cout << "Fazendo t" << std::endl;


    for (int i = 0; i < total_pixels; i++) {
        image_t->SetPixel({mapa.at(i).first, mapa.at(i).second}, t(i,0));
    }

    /*
    c = 0;
    ptr = image_t->GetBufferPointer();
    while (c < total_pixels) {
        *(ptr+c) = t(c,0);
        c++;
    }
    */

    std::cout << " feito" << std::endl;
//    L.print(std::cout);
}


arma::mat achaW (ImageType::Pointer image_in, int x, int y) {
    arma::mat W(9,3);
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            int px = std::min(std::max(i+x-1,0), largura-1);
            int py = std::min(std::max(j+x-1,0), altura-1);
            auto pixel = image_in->GetPixel({px,py});
            W(j*3+i,0) = pixel[0];
            W(j*3+i,1) = pixel[1];
            W(j*3+i,2) = pixel[2];
        }
    }
    return (W);
}

double Lij(ImageType::Pointer image_in, int ix, int iy, int jx, int jy, std::map<unsigned int, arma::mat> const& Wcov, std::map<unsigned int, arma::mat> const& Wmean) {
    arma::mat I_i(3,1), I_j(3,1);
    {
        auto Pi = image_in->GetPixel({ix,iy});
        I_i(0,0) = Pi[0];
        I_i(1,0) = Pi[1];
        I_i(2,0) = Pi[2];
        auto Pj = image_in->GetPixel({jx,jy});
        I_j(0,0) = Pj[0];
        I_j(1,0) = Pj[1];
        I_j(2,0) = Pj[2];
    }
    double retorno = 0.0;

//    std::cout << "LIJ ix = " << ix << " iy = " << iy << " jx = " << jx << " jy = " << jy << " size: " << size[0] << "x" << size[1] << std::endl;

    /*     ix e iy   sao as coordenadas do ponto I_i
     *     jx e jy   sao as coordenadas do ponto I_j
     *     x e y sao as coordenadas da janela W
     */
    for (int x=std::max(ix-2,0); x<std::min(jx+3,(int) size[0]); x++) {
        for (int y=std::max(iy-2,0); y<std::min(jy+3, (int) size[1]); y++) {
//            std::cout << "Janela: x = " << x << " ix = " << ix << " jx = " << jx << "   y = " << y << " iy = " << iy << " jy = " << jy << std::endl;
//            std::cout << "Testando ponto " << x << "," << y << std::endl;
            if ( checaDistancia(x,y,ix,iy,2) && checaDistancia(x,y,jx,jy,2)) {
//                std::cout << "Janela em  " << x << "," << y << "  contem os dois pontos" << std::endl;
                int pos = mapa_inv[{x,y}];

                arma::mat tmp = I_i - Wmean.at(pos);
//                                tmp.print(std::cout, "TMP");
                arma::mat tmp2 = Wcov.at(pos) + (epsilon/(double)wk)*arma::eye(3,3);
 //                               tmp2.print(std::cout, "TMP2");
                arma::mat tmp3 = I_j - Wmean.at(pos);
  //                              tmp3.print(std::cout, "TMP3");
                arma::mat ltmp = tmp.t() * arma::inv(tmp2) * tmp3;
   //                             ltmp.print(std::cout, "Ltmp");
                retorno += ( -1.0/(double) wk)*(1+ltmp(0,0));

            }
        }
    }
    return retorno;
}

double laplacian(ImageType::Pointer image_in, unsigned int ix, unsigned int iy, unsigned int jx, unsigned int jy) {
    if (ix > jx) std::swap(ix, jx);
    if (iy > jy) std::swap(iy, jy);
    unsigned int wk = (jx-ix+1)*(jy-iy+1);
    std::cout << "wk=" << wk << std::endl;
    double retorno = 0.0;
    arma::mat W(wk,3);
    unsigned int c=0;
    for (unsigned int x=ix; x <= jx; x++) {
        for (unsigned int y=iy; y <= jy; y++) {
            auto pixel = image_in->GetPixel({x,y});
            W(c,0) = pixel[0];
            W(c,1) = pixel[1];
            W(c,2) = pixel[2];
            c++;
        }
    }

    auto Wcov = arma::cov(W,W);
    auto Wmean = arma::mean(W,0);
    std::cout << "Matrizes:" << std::endl;
    std::cout << "W:" << std::endl;
    W.print(std::cout);
    std::cout << "Wcov:" << std::endl;
    Wcov.print(std::cout);
    std::cout << "Wmean:" << std::endl;
    Wmean.print(std::cout);

    for (unsigned int x=ix; x <= jx; x++) {
        for (unsigned int y=iy; y <= jy; y++) {
            unsigned int c=0;
            for (unsigned int i1=x-1;i1<=x+1;i1++)
                for (unsigned int j1=y-1;j1<=y+1;j1++)
                    for (unsigned int i2=x-1;i2<=x+1;i2++)
                        for (unsigned int j2=y-1;j2<=y+1;j2++) {
                            auto Pi = image_in->GetPixel({i1, j1});
                            auto Pj = image_in->GetPixel({i2, j2});
                            arma::mat Ii(1,3), Ij(1,3);
                            Ii(0,0) = Pi[0];
                            Ii(0,1) = Pi[1];
                            Ii(0,2) = Pi[2];

                            Ij(0,0) = Pj[0];
                            Ij(0,1) = Pj[1];
                            Ij(0,2) = Pj[2];

                            auto tmp = Ii - Wmean;
                            //                std::cout << "Tmp: " << std::endl;
                            //                tmp.print(std::cout);
                            auto tmp2 = Wcov + (epsilon/(double)wk)*arma::eye(3,3);
                            //                std::cout << "Tmp2: " << std::endl;
                            //                tmp2.print(std::cout);
                            auto tmp3 = (Ij-Wmean);
                            //                std::cout << "Tmp3: " << std::endl;
                            //                tmp3.print(std::cout);

                            auto ltmp = tmp * tmp2 * arma::trans(tmp3);
                            //                std::cout << "Ltmp: " << std::endl;
                            //                ltmp.print(std::cout);
                            double ltmp2 = 1+arma::accu(ltmp);
                            retorno += ( (i1==i2 && j1==j2 ? 1 : 0) - (1.0/(double) wk)*(ltmp2));
                        }
        }
    }
    return retorno;
}

void thread_L(ImageType::Pointer image, int ix, int iy, int total_pixels, arma::sp_fmat &L, std::mutex &mtx, std::vector<Cache> const &cache) {
//    arma::sp_mat Ltmp(total_pixels, total_pixels);
    std::unordered_map<int, float> Ltmp;

    int i = ConverteXY2I(ix, iy);

    for (int jx = ix+1; jx < std::min(ix+3,(int) size[0]); jx++) {
            double tmp = matting_L(image, ix, iy, jx, iy, cache);
            if (!std::isnan(tmp)) {
                //Ltmp[j] = tmp;
                Ltmp[ConverteXY2I(jx,iy)] = tmp;
            }
    }

    if (iy+1 < size[1]) 
    for (int jy = iy+1; jy < std::min(iy+3,(int) size[1]); jy++) {
        for (int jx = std::max(ix-2,0); jx < std::min(ix+3,(int) size[0]); jx++) {
                double tmp = matting_L(image, ix, iy, jx, jy, cache);
                if (!std::isnan(tmp)) {
                    //Ltmp[j] = tmp;
                    Ltmp[ConverteXY2I(jx,jy)] = tmp;
                }
        }
    }
    {
        std::lock_guard<std::mutex> guard(mtx);
        for (auto &tmp : Ltmp) {
            L(i,tmp.first) = L(tmp.first,i) = tmp.second;
            //L(i,tmp.first) = tmp.second;
        }
    }
    posts--;
}

void matting2 (ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t) {

    long int total_pixels = size[0]*size[1];
    std::cout << "matting2: " << size[0] << "x" << size[1] << std::endl;
    arma::sp_fmat L(total_pixels, total_pixels);

    std::vector<Cache> cache;
    cache.reserve(total_pixels);


    std::cout << "Fazendo cache..." << std::flush;
    {
        int w = 0;
        for (int y=0; y<size[1]; y++) {
            for (int x=0; x<size[0]; x++) {
                Cache tmp;
                arma::fmat W(carregaJanela(image_in, x, y));
                tmp.mean = arma::mean(W).t();
                arma::fmat Wcov(W.t()*W/wk - tmp.mean*tmp.mean.t());
                tmp.cov_inv = arma::inv(Wcov + (float) (epsilon/wk)*arma::eye<arma::fmat>(3,3));
                cache[w++] = tmp;
            }
        }
    }
    std::cout << " feito" << std::endl;

    boost::asio::io_service _io;
    std::unique_ptr<boost::asio::io_service::work> _work(
                new boost::asio::io_service::work(_io));
    boost::thread_group _threads;
    for (int i=0; i < THREADS; i++) {
        _threads.add_thread(new boost::thread(boost::bind(&boost::asio::io_service::run, &_io)));
    }
    std::mutex mtx;
    posts=0;

    std::cout << "Fazendo L..." << std::endl;
    for (int iy=0; iy < size[1]; iy++) {
        if (iy % 1 == 0) std::cout << "   -> " << iy << " de " << size[1] << std::endl;
        for (int ix=0; ix < size[0]; ix++) {
            //        for (int j=i+1; j < total_pixels; j++) {
            //            std::cout << "      -> Posting L(" << i << ")" << std::flush;
            _io.post(boost::bind(&thread_L, image_in, ix, iy, total_pixels, boost::ref(L), boost::ref(mtx), boost::ref(cache)));
            posts++;
            //            std::cout << " done" << std::endl;
            /*
               double tmp = matting_L (image_in, i, j);
               if (!std::isnan(tmp)) {
               L(j,i) = L(i,j) = tmp;
               soma += tmp;
               }
               */
            //        }
        }
    }

    std::cout << "All processes posted to threads, joining..." << std::endl;
    std::cout << "posts: " << posts << std::endl;
    while (posts>0) {
        std::cout << "posts: " << posts << std::endl;
        usleep(1e6);
    }
    _work.reset();
    _threads.join_all();
    std::cout << "All threads joined" << std::endl;
    std::vector<Cache>().swap(cache);
    std::cout << "Cache cleared" << std::endl;
    std::cout << "Making symmetric matrix... " << std::flush;
    //L = arma::symmatu(L);
    std::cout << "done" << std::endl;

    {
        std::cout << "Fazendo diagonal... " << std::flush;
        auto Lsum = arma::sum(L,1);
        std::cout << " Lsum" << std::flush;
        L -= diagmat(Lsum);
        std::cout << " lambda" << std::flush;
        L.diag() += lambda;
        std::cout << " feito" << std::endl;
    }

    /*
    for (int i=0; i<total_pixels; i++) {
        double soma = 0;
        for (int j=0; j<total_pixels; j++) {
            if (j!=i) soma+= L(i,j);
        }
        L(i,i) = -soma;
    }
    */

    std::cout << "Checando L..." << std::endl;
//    checaL(L);
    
    std::cout << "Somando Lambda..." << std::endl;
//    L.diag() += lambda;

    std::cout << "Fazendo Mtchapeu..." << std::endl;
    arma::fmat Mtchapeu(total_pixels,1);
    {
        auto *ptr = image_tchapeu->GetBufferPointer();
        unsigned int c = 0;
        double tmin=1.0, tmax=0.0;
        while (c < total_pixels) {
            if (*(ptr+c) > tmax) tmax = *(ptr+c);
            if (*(ptr+c) < tmin) tmin = *(ptr+c);
            Mtchapeu(c,0) = *(ptr+c)*lambda;
            c++;
        }
        std::cout << "*******************************************************" << std::endl;
        std::cout << "*******************************************************" << std::endl;
        std::cout << "tcmin = " << tmin << "  tcmax = " << tmax << std::endl;
        std::cout << "*******************************************************" << std::endl;
        std::cout << "*******************************************************" << std::endl;

    }
    std::cout << "Fazendo Mt..." << std::endl;
    arma::fmat Mt;
    arma::superlu_opts opts;
    opts.symmetric=true;
    opts.refine=arma::superlu_opts::REF_NONE;
    arma::spsolve(Mt, L, Mtchapeu, "superlu", opts);
    std::cout << "Fazendo t..." << std::endl;
    {
        double tmin=Mt(0,0), tmax=Mt(0,0);
        auto *ptr = image_t->GetBufferPointer();
        unsigned int c = 0;
        while (c < total_pixels) {
            *(ptr+c) = Mt(c,0);
            if (DEBUG || VERBOSE) std::cout << "  -> t(" << c << ") = " << Mt(c,0) << "    tchapeu = " << Mtchapeu(c,0) << std::endl;
            if (Mt(c,0) > tmax) tmax = Mt(c,0);
            if (Mt(c,0) < tmin) tmin = Mt(c,0);
            c++;
        }
        std::cout << "*******************************************************" << std::endl;
        std::cout << "*******************************************************" << std::endl;
        std::cout << "tmin = " << tmin << "  tmax = " << tmax << std::endl;
        std::cout << "*******************************************************" << std::endl;
        std::cout << "*******************************************************" << std::endl;
        /*
        c=0;
        while (c < total_pixels) {
            *(ptr+c) = (*(ptr+c)-tmin) * (tmax-tmin);
            c++;
        }
        */
    }
}

double matting_L (ImageType::Pointer image, int ix, int iy, int jx, int jy, std::vector<Cache> const &cache) {
    std::vector<std::pair<int, int>> Ws(achaJanelas (ix, iy, jx, jy));
    if (Ws.size() == 0) return std::numeric_limits<double>::quiet_NaN();
    double soma = 0.0;
    for (auto &w : Ws) {
        soma += calculaL (image, ix, iy, jx, jy, w, cache);
    }
    return soma;
}

std::vector<std::pair<int, int>> achaJanelas (int ix, int iy, int jx, int jy) {
    //    int ix, iy, jx, jy;
    //    std::tie(ix,iy) = ConverteI2XY(i);
    //    std::tie(jx,jy) = ConverteI2XY(j);



    /*
       int ix = mapa[i].first,
       iy = mapa[i].second,
       jx = mapa[j].first,
       jy = mapa[j].second;
       */
    std::vector<std::pair<int, int>> ws;

    if (abs(ix-jx) < 3 && abs(iy-jy) < 3) {
        int xmin = std::max(std::min(ix,jx)-2,0),
            xmax = std::min(std::max(ix,jx)+2,largura-1),
            ymin = std::max(std::min(iy,jy)-2,0),
            ymax = std::min(std::max(iy,jy)+2,altura-1);

        for (int x = xmin; x <= xmax; x++) {
            for (int y = ymin; y <= ymax; y++) {
                if (    abs(x-ix) < 2 &&
                        abs(x-jx) < 2 &&
                        abs(y-iy) < 2 &&
                        abs(y-jy) < 2) {
                    ws.push_back({x,y});
                }
            }
        }
    }
    return ws;
}

arma::fmat carregaJanela (ImageType::Pointer image, int x, int y) {
//    int x,y;
//    std::tie(x,y) = ConverteI2XY(w);
//    int x = mapa[w].first, y = mapa[w].second;
    arma::fmat W(9,3);
    ImageType::SizeType size;
    size = image->GetLargestPossibleRegion().GetSize();
    unsigned int c = 0;
    for (int i = x-1; i <= x+1; i++) {
        for (int j = y-1; j <= y+1; j++) {
            int px = i, py = j;
            if (px < 0) px = 0;
             else if (px >= size[0]) px = size[0]-1;
            if (py < 0) py = 0;
             else if (py >= size[1]) py = size[1]-1;
            auto pixel = image->GetPixel({px, py});
            W(c,0) = pixel[0];
            W(c,1) = pixel[1];
            W(c,2) = pixel[2];
            c++;
            if (DEBUG) {
                std::cout << "carregaJanela: (" << i << "," << j << ")->pixel(" << px << "," << py << ") = " << pixel[0] << " " << pixel[1] << " " << pixel[2] << std::endl;

            }
        }
    }
    return W;
}

double calculaL (ImageType::Pointer image, int ix, int iy, int jx, int jy, std::pair<int, int> wp, std::vector<Cache> const &cache) {
    using namespace std;
    int w = ConverteXY2I(wp.first, wp.second);
    if (DEBUG) std::cout << "calculaL i = (" << ix << "," << iy << ") j = (" << jx << "," << jy << ") w = ( " << wp.first << "," << wp.second << " )" << std::endl;
    double resultado = (ix==jx && iy == jy) ? 1.0 : 0.0;
    auto pixelI = image->GetPixel({ix, iy});
    auto pixelJ = image->GetPixel({jx, jy});
    arma::mat Ii, Ij;
    Ii << pixelI[0] << arma::endr << pixelI[1] << arma::endr << pixelI[2];
    Ij << pixelJ[0] << arma::endr << pixelJ[1] << arma::endr << pixelJ[2];
    if (DEBUG) {
        Ii.print(cout, "Ii");
        Ij.print(cout, "Ij");
    }

    arma::mat tmp = (Ii - cache[w].mean).t() * cache[w].cov_inv * (Ij - cache[w].mean);
    if (DEBUG) tmp.print(cout, "tmp");
    resultado -= (1+tmp(0,0))/wk;
    return resultado;
}


void guided_filtering (ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t) {
    std::cout << "Starting Guided Filtering" << std::endl;
    arma::fmat  I(altura, largura);
    arma::fmat  P(altura, largura);
    std::cout << "Steps: " << std::endl;
    for (int i=0; i < altura; i++) {
        for (int j=0; j < largura; j++) {
            I(i,j) = image_in->GetPixel({j,i}).GetLuminance();
            P(i,j) = image_tchapeu->GetPixel({j,i});
        }
    }
    if (DEBUG) {
        I.print(std::cout, "I");
        P.print(std::cout, "P");
    }

#if 1
    std::cout << " -> Loaded I,P" << std::endl;
    arma::fmat meanI(fmean(I));
    std::cout << " -> fmean(I)" << std::endl;
    arma::fmat meanP = fmean(P);
    std::cout << " -> fmean(P)" << std::endl;
    arma::fmat corrI = fmean(I % I);
    std::cout << " -> corr(I,I)" << std::endl;
    arma::fmat corrIP = fmean (I % P);
    std::cout << " -> corr(I,P)" << std::endl;
    arma::fmat varI = corrI - (meanI % meanI);
    std::cout << " -> var(I)" << std::endl;
    arma::fmat covIP = corrIP - (meanI % meanP);
    std::cout << " -> cov(I,P)" << std::endl;
    arma::fmat a = covIP / (varI + epsilonG);
    std::cout << " -> a" << std::endl;
    arma::fmat b = meanP - a % meanI;
    std::cout << " -> b" << std::endl;
    arma::fmat meanA = fmean(a);
    std::cout << " -> fmean(a)" << std::endl;
    arma::fmat meanB = fmean(b);
    std::cout << " -> fmean(b)" << std::endl;
    arma::fmat q = (meanA % I) + meanB;
    std::cout << " -> q" << std::endl;


    if (DEBUG) {
        meanI.print(std::cout, "meanI");
        meanP.print(std::cout, "meanP");
    }
#else
    arma::fmat q = fmean(P);
#endif

    std::cout << "Loading image_t" << std::endl;

    for (int i=0; i < altura; i++) {
        for (int j=0; j < largura; j++) {
            image_t->SetPixel({j,i}, q(i,j));
            if (VERBOSE) {
                std::cout << "\t-> Ponto (" << i << "," << j << "):  " << P(i,j) << " => " << q(i,j) << std::endl;
            }
        }
    }
    std::cout << "Done" << std::endl;
}

void sumRow(arma::fmat const &I_in, int r, arma::fmat &retorno, int i, std::mutex &mtx) {
    arma::Row<float> a = I_in.row(i);
    arma::Row<float> tmp(a.n_elem); 
    for (int i=0; i < a.n_elem; i++) {
        int b = std::max(0,i-r/2);
        int e = std::min(i+r/2,(int) a.n_elem-1);
        tmp(i) = (arma::mean(a(arma::span(b,e))));
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        retorno.row(i) = tmp;
    }
}

void sumCol(arma::fmat const &I_in, int r, arma::fmat &retorno, int i, std::mutex &mtx) {
    arma::Col<float> a = I_in.col(i);
    arma::Col<float> tmp(a.n_elem); 
    for (int i=0; i < a.n_elem; i++) {
        int b = std::max(0,i-r/2);
        int e = std::min(i+r/2,(int) a.n_elem-1);
        tmp(i) = (arma::mean(a(arma::span(b,e))));
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        retorno.col(i) = tmp;
    }
}



arma::fmat fmean(arma::fmat const &I_in) {
//    return fmean2(I_in);
    int r = 200;
    arma::fmat retorno(arma::size(I_in));
    arma::fmat retorno2(arma::size(I_in));
    std::vector<std::future<void>> jobs;
    std::mutex mtx;

    for (int i=0; i<I_in.n_rows; i++) {
        jobs.push_back(std::async(std::launch::async, sumRow, std::cref(I_in), r, std::ref(retorno), i, std::ref(mtx)));
    }
    for (auto &i : jobs) i.wait();
    std::vector<std::future<void>>().swap(jobs);

    for (int i=0; i<I_in.n_cols; i++) {
        jobs.push_back(std::async(std::launch::async, sumCol, std::cref(retorno), r, std::ref(retorno2), i, std::ref(mtx)));
    }
    for (auto &i : jobs) i.wait();

    return retorno2;
}

arma::fmat fmean2(arma::fmat const &I) {
    int r = 60;
    //arma::fmat tmp(I.n_rows, I.n_cols);
    //arma::fmat tmp2(arma::size(I));
    arma::fmat Mconv(2*r,2*r);
    Mconv.ones();
    Mconv /= (4*r*r);
//    arma::fmat Mconv = arma::ones(2*r,2*r)/(4*r*r);
    arma::fmat tmp2 = arma::conv2(I, Mconv, "same");

    /*
    for (int y=0; y < I.n_rows; y++) {
//        if (y%100==0) std::cout << " -> " << y << "/" << I.n_rows << std::endl;
        for (int x=0; x < I.n_cols; x++) {
            *//*
               arma::fmat W = fetchWindow(I, x, y, r);
               tmp2(x,y) = arma::accu(W);
               */
    /*
            tmp2(y,x) = fetchMean(I, x, y, r);
        }
    }
*/
    //    tmp2 = tmp2/(double) (r*r);
    return tmp2;
}

float fetchMean(arma::fmat const &I, int x, int y, int r) {
    if (VERBOSE) {
        std::cout << "fetchWindow: x=" << x << " y=" << y << " r=" << r << std::endl;
    }
    int r2 = r/2;

    int xi = std::max(x-r2, 0);
    int yi = std::max(y-r2, 0);
    int xf = std::min(x+r2, ((int) I.n_cols)-1);
    int yf = std::min(y+r2, ((int) I.n_rows)-1);

    arma::fmat W = I(arma::span(yi,yf), arma::span(xi,xf));
    float mean = arma::accu(W)/(W.n_cols*W.n_rows);

    if (VERBOSE) {
        std::cout << "x = [" << xi << "," << xf << "]  y = [" << yi << "," << yf << "]" << std::endl;
        W.print(std::cout, "W");
    }
    return mean;
}

arma::fmat fetchWindow(arma::fmat const &I, int x, int y, int r) {
    if (VERBOSE) {
        std::cout << "fetchWindow: x=" << x << " y=" << y << " r=" << r << std::endl;
    }
    int r2 = r/2;
    arma::fmat tmp(r, r);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            int px = std::min((int) I.n_cols-1, std::max(0, x+i-r2));
            int py = std::min((int) I.n_rows-1, std::max(0, y+j-r2));
            tmp (i,j) = I(px,py);
        }
    }
//    tmp.print(std::cout, "fetchWindow");
    return tmp;
}
