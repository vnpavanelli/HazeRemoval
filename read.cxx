#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "QuickView.h"
#include <armadillo>
#include <stdlib.h>
 
  typedef float PixelComponent;
  typedef itk::RGBPixel< PixelComponent >    PixelType;
  typedef itk::Image<PixelType, 2> ImageType;
  typedef itk::Image<PixelComponent, 2> ImageGrayType;

  const double epsilon = 1e-3;
  const unsigned int wk = 9;


template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);
void minima(typename ImageType::Pointer image_in, typename ImageGrayType::Pointer image_min);
void dark_channel(typename ImageGrayType::Pointer image_min, typename ImageGrayType::Pointer image_dark);
double achaA(typename ImageGrayType::Pointer image_dark, typename ImageType::Pointer image_in);
void tiraHaze(ImageType::Pointer image_in, ImageGrayType::Pointer image_gray, ImageType::Pointer image_out, double pixel_A);
void corrigeA(ImageType::Pointer image_in, double A);
void matting(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t);
double laplacian(ImageType::Pointer image_in, unsigned int ix, unsigned int iy, unsigned int jx, unsigned int jy);
double Lij(ImageType::Pointer image_in, int ix, int iy, int jx, int jy);
void calculaT(typename ImageGrayType::Pointer image_dark, typename ImageGrayType::Pointer image_t);


void uchar2float(typename ImageType::Pointer image, typename ImageType::Pointer image_in);
 
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
  ImageType::Pointer image_out = ImageType::New();
  ImageGrayType::Pointer image_min = ImageGrayType::New();
  ImageGrayType::Pointer image_dark = ImageGrayType::New();
  ImageGrayType::Pointer image_tchapeu = ImageGrayType::New();
  ImageGrayType::Pointer image_t = ImageGrayType::New();

  ReadFile<ImageType>(inputFilename, image);
  
  ImageGrayType::RegionType region;
  ImageGrayType::IndexType start;
  ImageGrayType::SizeType size;
  start[0] = start[1] = 0;
  size = image->GetLargestPossibleRegion().GetSize();
  region.SetSize(size);
  region.SetIndex(start);
  image_dark->SetRegions(region);
  image_dark->Allocate();

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

  uchar2float (image, image_in);
  minima (image_in, image_min);
  dark_channel (image_min, image_dark);
  double pixel_A = achaA(image_dark, image_in);
  std::cout << "Valor de A atmosferico: " << pixel_A << std::endl;


  corrigeA (image_in, pixel_A);
  minima (image_in, image_min);
  dark_channel (image_min, image_dark);
  calculaT (image_dark, image_tchapeu);
  matting (image_in, image_tchapeu, image_t);

  tiraHaze(image_in, image_dark, image_out, pixel_A);

  std::cout << "Exibindo imagens" << std::endl;

  QuickView viewer;
  viewer.AddImage(image_in.GetPointer());
  viewer.AddImage(image_t.GetPointer());
  viewer.AddImage(image_tchapeu.GetPointer());
  viewer.AddImage(image_out.GetPointer());
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
  const unsigned int Patch_window = 15;

  std::cout << "Criando Dark Channel..." << std::endl;
  /* Criar o Prior Dark Channel usando o minimo de um filtro Patch_window X Patch_window (e.g. 15x15) */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto min = image_min->GetPixel({i,j});
          unsigned int i1, i2, j1, j2;
          i1 = (i < Patch_window) ? 0 : i-Patch_window;
          i2 = (i+Patch_window >= size[0]) ? size[0] : i+Patch_window;
          j1 = (j < Patch_window) ? 0 : j-Patch_window;
          j2 = (j+Patch_window >= size[1]) ? size[1] : j+Patch_window;
          for (;i1<i2;i1++) {
              for (;j1<j2;j1++) {
                  min = std::min(min, image_min->GetPixel({i1, j1}));
              }
          }
          image_dark->SetPixel({i,j}, min);
      }
  }
}


double achaA(typename ImageGrayType::Pointer image_dark, typename ImageType::Pointer image_in) {
  ImageType::SizeType size;
  size = image_in->GetLargestPossibleRegion().GetSize();

  std::cout << "Procurando valor de A..." << std::endl;
  unsigned int total_pixels = size[0]*size[1];
  int porcento01 = ceil((double) total_pixels * 0.001);
  float pixel_max = 1.0f;
  std::cout << "Total pixels: " << total_pixels << std::endl;
  std::cout << "0.1%: " << porcento01 << std::endl;
  PixelType::LuminanceType pixel_A=0;
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
                  pixel_A = std::max(pixel_A, image_in->GetPixel({i,j}).GetLuminance());
              }
          }
      }
      pixel_max-=.01;
  } while (porcento01>0 && pixel_max >0);
  return pixel_A;
}


void tiraHaze(ImageType::Pointer image_in, ImageGrayType::Pointer image_dark, ImageType::Pointer image_out, double pixel_A) {
  ImageType::SizeType size;
  size = image_in->GetLargestPossibleRegion().GetSize();

  PixelComponent A = pixel_A;
  PixelComponent t_max = 0.1;
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          PixelType pixel = image_in->GetPixel({i,j});
          PixelType pixelout;
          PixelComponent t_dark = 1.0-0.95*(image_dark->GetPixel({i,j}));
          double t = std::max(t_max,t_dark);
          pixelout[0] = (pixel[0] - A)/t + A; 
          pixelout[1] = (pixel[1] - A)/t + A; 
          pixelout[2] = (pixel[2] - A)/t + A; 
          /*
          std::cout << " -> (" << i << "," << j << ") t: " << (int) t << " pixel0: " << (int) pixel[0] << " pixelout0: " << (int) pixelout[0] 
               << "P-A:" << (int) pixel[0]-A << " P-A/t: " <<(int)  (pixel[0]-A)/t
              << std::endl;
              */
          image_out->SetPixel({i,j}, pixelout);
      }
  }
}

void corrigeA(ImageType::Pointer image_in, double A) {
  ImageType::SizeType size;
  size = image_in->GetLargestPossibleRegion().GetSize();

  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image_in->GetPixel({i,j});
          pixel[0] /= A;
          pixel[1] /= A;
          pixel[2] /= A;
          image_in->SetPixel({i,j}, pixel);
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
//        cout << "Fazendo L(" << i << "," << i << ")" << endl;
        double soma=0.0;
        for (int j=0; j < L.n_cols; j++) {
            if (i!=j) {
//                cout << "Somando " << L(i,j) << endl;
                soma += L(i,j);
            }
        }
//        cout << "L(" << i << "," << i << ") = " << L(i,i) << " == " << soma << endl;
        L(i,i) = soma;
    }
}

void matting(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t) {
    ImageType::SizeType size;
    size = image_in->GetLargestPossibleRegion().GetSize();
    unsigned largura=5, altura=5;
//    size[0] = 4; size[1] = 4;
    unsigned int total_pixels = size[0]*size[1];
    const unsigned int wk = 9;
    const double lambda = 1e-4;
    std::cout << "Total pixels: " << total_pixels << std::endl;
    arma::sp_mat L(total_pixels, total_pixels);

    std::cout << "Laplacian: " << Lij (image_in, 0, 0, 1, 1) << std::endl;
    std::cout << "Preenchendo L: " << std::flush;

//                exit(0);
    for (int i1 = 0; i1 < size[0]; i1++) {
        std::cout << " -> " << i1 << " de " << size[0] << std::endl;
        for (int j1=0; j1 < size[1]; j1++) {
            for (int i2 = std::max(i1-2,0) ; i2 < std::min(i1+3,(int) size[0]) ; i2++) {
                for (int j2 = std::max(j1-2,0) ; j2 < std::min(j1+3,(int) size[1]) ; j2++) {
                    //std::cout << "    -> Ponto (" << i1 << "," << j1 << ") e (" << i2 << "," << j2 << ")" << std::endl;
                    if (abs(i1-i2)<=2 && abs(j1-j2)<=2 && i2>=0 && j2 >= 0) {
//                        std::cout << "    -> Ponto (" << i1 << "," << j1 << ") e (" << i2 << "," << j2 << ")" << std::endl;
                        double tmp = Lij(image_in, i1, j1, i2, j2);
//                        std::cout << "        -> L (" << i1*size[0]+j1 << "," << i2*size[0]+j2 << ") = " << tmp << std::endl;
                        L(i1*size[0]+j1, i2*size[0]+j2) = tmp;
                    }

                }
            }
        }
    }

    std::cout << "L preenchida" << std::endl;
//    L.print(std::cout);
    checaL(L);
    std::cout << "Fazendo L + lambda" << std::endl;

    L += arma::eye(total_pixels, total_pixels)*lambda;
    //L = arma::eye(total_pixels, total_pixels)*2e-4;

    std::cout << "Fazendo tchapeu" << std::endl;
    arma::mat tchapeu(total_pixels,1);
    unsigned int c=0;
    auto *ptr = image_tchapeu->GetBufferPointer();
    while (c < total_pixels) {
        tchapeu(c,0) = *(ptr+c) * lambda;
        c++;
    }

    std::cout << "Fazendo spsolve" << std::endl;
    auto t = arma::spsolve(L,tchapeu);

    std::cout << "Fazendo t" << std::endl;
    c = 0;
    ptr = image_t->GetBufferPointer();
    while (c < total_pixels) {
        *(ptr+c) = t(c,0);
        c++;
    }

    std::cout << " feito" << std::endl;
//    L.print(std::cout);
}

double Lij(ImageType::Pointer image_in, int ix, int iy, int jx, int jy) {

    ImageType::SizeType size;
    size = image_in->GetLargestPossibleRegion().GetSize();


//    if (abs(ix-jx)>2 || abs(iy-jy)>2) return 0.0;
    if (ix > jx) std::swap(ix, jx);
    if (iy > jy) std::swap(iy, jy);

    arma::mat I_i(1,3), I_j(1,3);
    {
        auto Pi = image_in->GetPixel({ix,iy});
        I_i(0,0) = Pi[0];
        I_i(0,1) = Pi[1];
        I_i(0,2) = Pi[2];
        auto Pj = image_in->GetPixel({jx,jy});
        I_j(0,0) = Pj[0];
        I_j(0,1) = Pj[1];
        I_j(0,2) = Pj[2];
    }
    double retorno = 0.0;

    for (unsigned int x=ix; x<=jx+2; x++) {
        for (unsigned int y=iy; y<=jy+2; y++) {
//            std::cout << "Testando ponto " << x << "," << y << std::endl;
            if (    (abs(x-ix)<2 && abs(y-iy)<2)    &&    (abs(x-jx)<2 && abs(y-jy)<2) && x >= 0 && y >= 0  ) {
//                std::cout << "Janela em  " << x << "," << y << "  contem os dois pontos" << std::endl;

                /* Preenche janela W */
                arma::mat W(9,3);
                for (unsigned int i=0; i<3; i++) {
                    for (unsigned int j=0; j<3; j++) {
                        int px = abs(i+x-1), py = abs(j+y-1);
                        if (px >= size[0]) px = size[0]-1;
                        if (py >= size[1]) py = size[1]-1;
                        auto pixel = image_in->GetPixel({px,py});
                        W(i*3+j,0) = pixel[0];
                        W(i*3+j,1) = pixel[1];
                        W(i*3+j,2) = pixel[2];
                    }
                }



                /*
                                std::cout << "W: " << std::endl;
                                W.print(std::cout);
                                */

                /* Calcula covariancia e media */
                auto Wcov = arma::cov(W,W);
                /*
                                std::cout << "Wcov: " << std::endl;
                                Wcov.print(std::cout);
                                */
                auto Wmean = arma::mean(W,0);
                /*
                                std::cout << "Wmean: " << std::endl;
                                Wmean.print(std::cout);
                                */

                auto tmp = I_i - Wmean;
                /*
                                std::cout << "Tmp: " << std::endl;
                                tmp.print(std::cout);
                                */
                auto tmp2 = Wcov.t() + (epsilon/(double)wk)*arma::eye(3,3);
                /*
                                std::cout << "Tmp2: " << std::endl;
                                tmp2.print(std::cout);
                                */
                auto tmp3 = (I_j-Wmean);
                /*
                                std::cout << "Tmp3: " << std::endl;
                                tmp3.print(std::cout);
                                */

                auto ltmp = tmp * tmp2 * arma::trans(tmp3);
                /*
                                std::cout << "Ltmp: " << std::endl;
                                ltmp.print(std::cout);
                                */
                double ltmp2 = 1+arma::accu(ltmp);
                retorno += ( (ix==jx && iy==jy ? 1.0 : 0.0) - (1.0/(double) wk)*(ltmp2));
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



