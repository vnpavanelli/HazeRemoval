#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "QuickView.h"
#include <armadillo>
#include <stdlib.h>
#include <itkNeighborhood.h>
#include <itkNeighborhoodIterator.h>
#include <cmath>
#include <limits>
#include <math.h>
#include <tuple>
 
  typedef float PixelComponent;
  typedef itk::RGBPixel< PixelComponent >    PixelType;
  typedef itk::Image<PixelType, 2> ImageType;
  typedef itk::Image<PixelComponent, 2> ImageGrayType;

  const double epsilon = 1e-4; //1e-3;
  const double lambda = 1e-4;
//  const unsigned int wk = 9;
  const double wk = 9;

  const bool DEBUG = false;
  unsigned int largura = 0, altura = 0;

    std::map<unsigned int, std::pair<unsigned int, unsigned int>> mapa;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> mapa_inv;


template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);
void minima(typename ImageType::Pointer image_in, typename ImageGrayType::Pointer image_min);
void dark_channel(typename ImageGrayType::Pointer image_min, typename ImageGrayType::Pointer image_dark);
std::vector<double> achaA(typename ImageGrayType::Pointer image_dark, typename ImageType::Pointer image_in);
void tiraHaze(ImageType::Pointer image_in, ImageGrayType::Pointer image_gray, ImageType::Pointer image_out, std::vector<double> pixel_A);
void corrigeA(ImageType::Pointer image_in, ImageType::Pointer image_ac, std::vector<double> A);
void matting(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t);
void matting2(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t);
double laplacian(ImageType::Pointer image_in, unsigned int ix, unsigned int iy, unsigned int jx, unsigned int jy);
double Lij(ImageType::Pointer image_in, int ix, int iy, int jx, int jy, std::map<unsigned int, arma::mat> const&, std::map<unsigned int, arma::mat> const&);
void calculaT(typename ImageGrayType::Pointer image_dark, typename ImageGrayType::Pointer image_t);
arma::mat achaW (ImageType::Pointer image_in, int x, int y);
bool checaDistancia (int, int, int, int, unsigned int distancia=3);
bool checaDistancia (int, int, unsigned int distancia=3);


double calculaL (ImageType::Pointer image, int i, int j, int w);
std::vector<int> achaJanelas (int i, int j, ImageType::SizeType size);
double matting_L (ImageType::Pointer image, int i, int j);
arma::mat carregaJanela (ImageType::Pointer image, int w);

inline int ConverteXY2I (int, int);
inline std::pair<int, int> ConverteI2XY (int);
inline ImageType::IndexType ConverteI2Index (int);


void uchar2float(typename ImageType::Pointer image, typename ImageType::Pointer image_in);

int ConverteXY2I (int x, int y) {
    return (y*largura + x);
}

std::pair<int, int> ConverteI2XY (int i) {
    int x,y;
    y = i/largura;
    x = i % largura;
    return {x,y};
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
  std::string inputFilename = argv[1];
 

  ImageType::Pointer image = ImageType::New();
  ImageType::Pointer image_in = ImageType::New();
  ImageType::Pointer image_ac = ImageType::New();
  ImageType::Pointer image_out = ImageType::New();
  ImageGrayType::Pointer image_min = ImageGrayType::New();
  ImageGrayType::Pointer image_dark = ImageGrayType::New();
  ImageGrayType::Pointer image_tchapeu = ImageGrayType::New();
  ImageGrayType::Pointer image_t = ImageGrayType::New();
  ImageGrayType::Pointer image_t2 = ImageGrayType::New();

  ReadFile<ImageType>(inputFilename, image);
  
  ImageGrayType::RegionType region;
  ImageGrayType::IndexType start;
  ImageGrayType::SizeType size;
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
  image_dark->SetRegions(region);
  image_dark->Allocate();

  image_ac->SetRegions(region);
  image_ac->Allocate();

  image_min->SetRegions(region);
  image_min->Allocate();

  image_in->SetRegions(region);
  image_in->Allocate();

  image_out->SetRegions(region);
  image_out->Allocate();

  image_tchapeu->SetRegions(region);
  image_tchapeu->Allocate();

  image_t->SetRegions(region);
  image_t->Allocate();


  image_t2->SetRegions(region);
  image_t2->Allocate();

  uchar2float (image, image_in);
  minima (image_in, image_min);
  dark_channel (image_min, image_dark);
  std::vector<double> pixel_A = achaA(image_dark, image_in);
  std::cout << "Valor de A atmosferico: " << pixel_A[0] << "," << pixel_A[1] << "," << pixel_A[2] << std::endl;

  corrigeA (image_in, image_ac, pixel_A);
  minima (image_ac, image_min);
  dark_channel (image_min, image_dark);
  calculaT (image_dark, image_tchapeu);
//  matting (image_in, image_tchapeu, image_t);
  matting2 (image_in, image_tchapeu, image_t2);

  tiraHaze(image_in, image_t2, image_out, pixel_A);

  std::cout << "Exibindo imagens" << std::endl;

  QuickView viewer;
  viewer.AddImage(image_in.GetPointer(), true, "in");
  viewer.AddImage(image_min.GetPointer(), true, "min");
  viewer.AddImage(image_dark.GetPointer(), true, "dark");
  viewer.AddImage(image_t.GetPointer(), true, "t");
  viewer.AddImage(image_t2.GetPointer(), true, "t2");
  viewer.AddImage(image_tchapeu.GetPointer(), true, "tchapeu");
  viewer.AddImage(image_out.GetPointer(), true, "out");
  viewer.Visualize();

  return EXIT_SUCCESS;
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

void uchar2float(typename ImageType::Pointer image, typename ImageType::Pointer image_in) {
  ImageType::SizeType size;
  size = image->GetLargestPossibleRegion().GetSize();
  /* transforma imagem unsigned char em float */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image->GetPixel({i,j});
          pixel[0] /= 255.0;
          pixel[1] /= 255.0;
          pixel[2] /= 255.0;
          image_in->SetPixel({i, j}, pixel);
      }
  }
}

void minima(typename ImageType::Pointer image_in, typename ImageGrayType::Pointer image_min) {
  ImageType::SizeType size;
  size = image_in->GetLargestPossibleRegion().GetSize();

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
  ImageType::SizeType size;
  size = image_dark->GetLargestPossibleRegion().GetSize();
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          image_t->SetPixel({i,j}, 1.0 - 0.95*image_dark->GetPixel({i,j}));
      }
  }

}

void dark_channel(typename ImageGrayType::Pointer image_min, typename ImageGrayType::Pointer image_dark) {
  ImageType::SizeType size;
  size = image_min->GetLargestPossibleRegion().GetSize();
  const int Patch_window = 7;  // floor(15/2);

  std::cout << "Criando Dark Channel..." << std::endl;
  /* Criar o Prior Dark Channel usando o minimo de um filtro Patch_window X Patch_window (e.g. 15x15) */
  for (int i = 0; i < size[0]; i++) {
      for (int j = 0; j < size[1]; j++) {

          int i1, i2, j1, j2;
          i1 = std::max(i-Patch_window,0);
          i2 = std::min(i+Patch_window, (int) size[0]-1);
          j1 = std::max(j-Patch_window,0);
          j2 = std::min(j+Patch_window, (int) size[1]-1);
          auto min = image_min->GetPixel({i1,j1});
//          std::cout << "Dark channel para (" << i << "," << j << ") nos intervalos: (" << i1 << "," << j1 << ") e (" << i2 << "," << j2 << ")" << std::endl;
          for (;i1<=i2;i1++) {
//              std::cout << " i1 = " << i1 << "  i2 = " << i2 << std::endl;
              for (int ji = j1; ji<=j2;ji++) {
                  min = std::min(min, image_min->GetPixel({i1, ji}));
//                  std::cout << "  -> min = " << min << "  pixel = " << image_min->GetPixel({i1, ji}) << "  pos = (" << i1 << "," << ji << ")" << std::endl;
              }
          }
          image_dark->SetPixel({i,j}, min);
//          exit(1);
      }
  }
}


std::vector<double> achaA(typename ImageGrayType::Pointer image_dark, typename ImageType::Pointer image_in) {
  ImageType::SizeType size;
  size = image_in->GetLargestPossibleRegion().GetSize();

  std::cout << "Procurando valor de A..." << std::endl;
  unsigned int total_pixels = size[0]*size[1];
  int porcento01 = ceil((double) total_pixels * 0.001);
  float pixel_max = 1.0f;
  std::cout << "Total pixels: " << total_pixels << std::endl;
  std::cout << "0.1%: " << porcento01 << std::endl;
  std::vector<double> pixel_A;
  pixel_A.resize(3);
  do {
      for (unsigned int i = 0; i < size[0] && porcento01 > 0; i++) {
          for (unsigned int j = 0; j < size[1] && porcento01 > 0; j++) {
              auto pixel = image_dark->GetPixel({i,j});
              if (pixel >= pixel_max) {
                  /*
                  std::cout << " -> pixel: " << pixel << " >= " << pixel_max << "   porcento01:" << porcento01 << std::endl;
                  std::cout << "    Luminance: " << image->GetPixel({i,j}).GetLuminance() << std::endl;
                  */
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
  return pixel_A;
}


void tiraHaze(ImageType::Pointer image_in, ImageGrayType::Pointer image_dark, ImageType::Pointer image_out, std::vector<double> pixel_A) {
  ImageType::SizeType size;
  size = image_in->GetLargestPossibleRegion().GetSize();

  auto& A = pixel_A;
  PixelComponent t_max = 0.1;
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          PixelType pixel = image_in->GetPixel({i,j});
          PixelType pixelout;
          //PixelComponent t_dark = 1.0-0.95*(image_dark->GetPixel({i,j}));
          PixelComponent t_dark = image_dark->GetPixel({i,j});
          double t = std::max(t_max,t_dark);
          pixelout[0] = (pixel[0] - A[0])/t + A[0]; 
          pixelout[1] = (pixel[1] - A[1])/t + A[1]; 
          pixelout[2] = (pixel[2] - A[2])/t + A[2]; 
          if (DEBUG) {
          std::cout << " -> (" << i << "," << j << ") t: " << (double) t << " tx: " << t_dark <<  " pixel0: " << (double) pixel[0] << " pixelout0: " << (double) pixelout[0] 
               << " P-A:" << (double) pixel[0]-A[0] << " P-A/t: " <<(double)  (pixel[0]-A[0])/t
              << std::endl;
          }
          image_out->SetPixel({i,j}, pixelout);
      }
  }
}

void corrigeA(ImageType::Pointer image_in, ImageType::Pointer image_ac, std::vector<double> A) {
  ImageType::SizeType size;
  size = image_in->GetLargestPossibleRegion().GetSize();

  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image_in->GetPixel({i,j});
          pixel[0] /= A[0];
          pixel[1] /= A[1];
          pixel[2] /= A[2];
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
    ImageType::SizeType size;
    size = image_in->GetLargestPossibleRegion().GetSize();
//    size[0] = 4; size[1] = 4;
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

    if (DEBUG) {
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
    ImageType::SizeType size;
    size = image_in->GetLargestPossibleRegion().GetSize();
    int largura = size[0], altura = size[1];

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
    ImageType::SizeType size;
    size = image_in->GetLargestPossibleRegion().GetSize();
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



void matting2 (ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t) {

    ImageType::SizeType size;
    size = image_in->GetLargestPossibleRegion().GetSize();
    long int total_pixels = size[0]*size[1];
    std::cout << "matting2: " << size[0] << "x" << size[1] << std::endl;
    arma::sp_mat L(total_pixels, total_pixels);

    std::cout << "Fazendo L..." << std::endl;
    for (int i=0; i < total_pixels; i++) {
        if (i % 100 == 0) std::cout << "   -> " << i << " de " << total_pixels << std::endl;
        double soma = 0.0;
        for (int j=0; j < i; j++) soma += L(i,j);
        for (int j=i+1; j < total_pixels; j++) {
            double tmp = matting_L (image_in, i, j);
            if (!std::isnan(tmp)) {
                L(j,i) = L(i,j) = tmp;
                soma += tmp;
            }
        }
        L(i,i) = -soma;
    }

    std::cout << "Checando L..." << std::endl;
//    checaL(L);
    
    std::cout << "Somando Lambda..." << std::endl;
    for (int i=0; i < total_pixels; i++) L(i,i) += lambda;

    std::cout << "Fazendo Mtchapeu..." << std::endl;
    arma::mat Mtchapeu(total_pixels,1);
    {
        auto *ptr = image_tchapeu->GetBufferPointer();
        unsigned int c = 0;
        while (c < total_pixels) {
            Mtchapeu(c,0) = *(ptr+c)*lambda;
            c++;
        }
    }
    std::cout << "Fazendo Mt..." << std::endl;
    arma::mat Mt = arma::spsolve(L,Mtchapeu);
    std::cout << "Fazendo t..." << std::endl;
    {
        auto *ptr = image_t->GetBufferPointer();
        unsigned int c = 0;
        while (c < total_pixels) {
            *(ptr+c) = Mt(c,0);
            if (DEBUG) std::cout << "  -> t(" << c << ") = " << Mt(c,0) << "    tchapeu = " << Mtchapeu(c,0) << std::endl;
            c++;
        }
    }
}

double matting_L (ImageType::Pointer image, int i, int j) {
    ImageType::SizeType size;
    size = image->GetLargestPossibleRegion().GetSize();
    std::vector<int> Ws(achaJanelas (i, j, size));
    if (Ws.size() == 0) return std::numeric_limits<double>::quiet_NaN();
    double soma = 0.0;
    for (auto &w : Ws) {
        soma += calculaL (image, i, j, w);
    }
    return soma;
}

std::vector<int> achaJanelas (int i, int j, ImageType::SizeType size) {
    int ix, iy, jx, jy;
    std::tie(ix,iy) = ConverteI2XY(i);
    std::tie(jx,jy) = ConverteI2XY(j);
    /*
    int ix = mapa[i].first,
        iy = mapa[i].second,
        jx = mapa[j].first,
        jy = mapa[j].second;
    */
    std::vector<int> ws;
    int xmin = std::min(ix,jx)-2,
        xmax = std::max(ix,jx)+2,
        ymin = std::min(iy,jy)-2,
        ymax = std::max(iy,jy)+2;
    

    if (abs(ix-jx) < 3 && abs(iy-jy) < 3) {
        for (int x = xmin; x <= xmax; x++) {
            for (int y = ymin; y <= ymax; y++) {
                if (    abs(x-ix) < 2 &&
                        abs(x-jx) < 2 &&
                        abs(y-iy) < 2 &&
                        abs(y-jy) < 2) {
                    //ws.push_back(mapa_inv[{x,y}]);
                    ws.push_back(ConverteXY2I(x,y));
                }
            }
        }
    }
    return ws;
}

arma::mat carregaJanela (ImageType::Pointer image, int w) {
    int x,y;
    std::tie(x,y) = ConverteI2XY(w);
//    int x = mapa[w].first, y = mapa[w].second;
    arma::mat W(9,3);
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

double calculaL (ImageType::Pointer image, int i, int j, int w) {
    using namespace std;
//    static std::map<int, std::pair<arma::mat, arma::mat>> cache;
    if (DEBUG) std::cout << "calculaL i=" << i << " j=" << j << " w=" << w << std::endl;
    double resultado = (i==j) ? 1.0 : 0.0;
    /*
    auto pixelI = image->GetPixel({mapa[i].first, mapa[i].second});
    auto pixelJ = image->GetPixel({mapa[j].first, mapa[j].second});
    */
    auto pixelI = image->GetPixel(ConverteI2Index(i));
    auto pixelJ = image->GetPixel(ConverteI2Index(j));
    arma::mat Ii, Ij;
    Ii << pixelI[0] << arma::endr << pixelI[1] << arma::endr << pixelI[2];
    Ij << pixelJ[0] << arma::endr << pixelJ[1] << arma::endr << pixelJ[2];
//    arma::mat W, Wmean, Wcov;
    /*
    if (cache.count(w)) {
        Wmean = cache[w].first;
        Wcov = cache[w].second;
    } else {
    */
    arma::mat W(carregaJanela(image, w));
    arma::mat Wmean(arma::mean(W).t());
    arma::mat Wcov(arma::cov(W));
    //    cache[w] = {Wmean, Wcov};
    //}

    if (DEBUG) {
        Ii.print(cout, "Ii");
        Ij.print(cout, "Ij");
        W.print(cout, "W");
        Wmean.print(cout, "Wmean");
        Wcov.print(cout, "Wcov");
    }

    arma::mat tmp = (Ii - Wmean).t() * arma::inv ( Wcov + (epsilon/wk)*arma::eye(3,3)) * (Ij - Wmean);
    if (DEBUG) tmp.print(cout, "tmp");
    resultado -= (1+tmp(0,0))/wk;
    return resultado;
}
