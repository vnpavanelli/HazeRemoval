#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "QuickView.h"
#include <armadillo>
#include <stdlib.h>
#include <itkNeighborhood.h>
#include <itkNeighborhoodIterator.h>
 
  typedef float PixelComponent;
  typedef itk::RGBPixel< PixelComponent >    PixelType;
  typedef itk::Image<PixelType, 2> ImageType;
  typedef itk::Image<PixelComponent, 2> ImageGrayType;

  const double epsilon = 1e-4; //1e-3;
  const unsigned int wk = 9;


template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);
void minima(typename ImageType::Pointer image_in, typename ImageGrayType::Pointer image_min);
void dark_channel(typename ImageGrayType::Pointer image_min, typename ImageGrayType::Pointer image_dark);
std::vector<double> achaA(typename ImageGrayType::Pointer image_dark, typename ImageType::Pointer image_in);
void tiraHaze(ImageType::Pointer image_in, ImageGrayType::Pointer image_gray, ImageType::Pointer image_out, std::vector<double> pixel_A);
void corrigeA(ImageType::Pointer image_in, ImageType::Pointer image_ac, std::vector<double> A);
void matting(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t);
double laplacian(ImageType::Pointer image_in, unsigned int ix, unsigned int iy, unsigned int jx, unsigned int jy);
double Lij(ImageType::Pointer image_in, int ix, int iy, int jx, int jy, std::map<unsigned int, arma::mat> const&, std::map<unsigned int, arma::mat> const&);
void calculaT(typename ImageGrayType::Pointer image_dark, typename ImageGrayType::Pointer image_t);
arma::mat achaW (ImageType::Pointer image_in, int x, int y);


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
  ImageType::Pointer image_ac = ImageType::New();
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

  uchar2float (image, image_in);
  minima (image_in, image_min);
  dark_channel (image_min, image_dark);
  std::vector<double> pixel_A = achaA(image_dark, image_in);
  std::cout << "Valor de A atmosferico: " << pixel_A[0] << "," << pixel_A[1] << "," << pixel_A[2] << std::endl;

  corrigeA (image_in, image_ac, pixel_A);
  minima (image_ac, image_min);
  dark_channel (image_min, image_dark);
  calculaT (image_dark, image_tchapeu);
  //matting (image_in, image_tchapeu, image_t);

  tiraHaze(image_in, image_tchapeu, image_out, pixel_A);

  std::cout << "Exibindo imagens" << std::endl;

  QuickView viewer;
  viewer.AddImage(image_in.GetPointer(), true, "in");
  viewer.AddImage(image_min.GetPointer(), true, "min");
  viewer.AddImage(image_dark.GetPointer(), true, "dark");
  viewer.AddImage(image_t.GetPointer(), true, "t");
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
          /*
          std::cout << " -> (" << i << "," << j << ") t: " << (int) t << " pixel0: " << (int) pixel[0] << " pixelout0: " << (int) pixelout[0] 
               << "P-A:" << (int) pixel[0]-A << " P-A/t: " <<(int)  (pixel[0]-A)/t
              << std::endl;
              */
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
        cout << "Fazendo L(" << i << "," << i << ")" << endl;
        double soma=0.0;
        for (int j=0; j < L.n_cols; j++) {
            if (i!=j) {
                cout << "Somando " << L(i,j) << endl;
                soma += L(i,j);
            }
        }
        soma *= -1.0;
        cout << "L(" << i << "," << i << ") = " << L(i,i) << " == " << soma << endl;
        L(i,i) = soma;
    }
}

void matting(ImageType::Pointer image_in, ImageGrayType::Pointer image_tchapeu, ImageGrayType::Pointer image_t) {
    ImageType::SizeType size;
    size = image_in->GetLargestPossibleRegion().GetSize();
    unsigned largura=5, altura=5;
//    size[0] = 4; size[1] = 4;
    unsigned int total_pixels = size[0]*size[1];
    const double lambda = 1e-4;
    std::cout << "Total pixels: " << total_pixels << std::endl;

    std::map<unsigned int, std::pair<unsigned int, unsigned int>> mapa;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> mapa_inv;
    for (int i = 0; i < size[0]; i++) {
        for (int j = 0; j < size[1]; j++) {
            mapa[j*size[0]+i] = {i,j};
            mapa_inv[{i,j}] = j*size[0]+i;
        }
    }

    std::map<unsigned int, arma::mat> Wcov, Wmean;

    std::cout << "Fazendo Wcov e Wmean..." << std::endl;
    for (int i=0; i<size[0]; i++) {
        for (int j=0; j<size[1]; j++) {
            auto W = achaW(image_in, i, j);
            Wcov[mapa_inv[{i,j}]] = arma::cov(W);
            Wmean[mapa_inv[{i,j}]] = arma::mean(W).t();
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

        for (int x=mapa[i].first; x < std::min ((unsigned int) mapa[i].first+3, (unsigned int) size[0]); x++) {
            for (int y=mapa[i].second; y < std::min( (unsigned int) mapa[i].second+3, (unsigned int) size[1]); y++) {
                if (x != mapa[i].first || y != mapa[i].second) {
                    unsigned int j = mapa_inv[{x,y}];
                    double tmp = Lij(image_in, mapa[i].first, mapa[i].second, x, y, Wcov, Wmean);
                    L(i,j) = L(j,i) = tmp;
                    soma += tmp;
                }
            }
        }
        L(i,i) = -1.0 * soma + lambda;
    }

    std::cout << "L preenchida" << std::endl;
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
    unsigned int c=0;
    auto *ptr = image_tchapeu->GetBufferPointer();
    while (c < total_pixels) {
        tchapeu(c,0) = *(ptr+c) * lambda;
        c++;
    }

    std::cout << "tchapeu: " << tchapeu(0,0) << " " << tchapeu(1,0) << " " << tchapeu(2,0) << std::endl;

    std::cout << "Fazendo spsolve" << std::endl;
    auto t = arma::spsolve(L,tchapeu);

    std::cout << "t: " << t(0,0) << " " << t(1,0) << " " << t(2,0) << std::endl;

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

//    std::cout << " ix = " << ix << " jx = " << jx << " iy = " << iy << " jy = " << jy << std::endl;

    /*     ix e iy   sao as coordenadas do ponto I_i
     *     jx e jy   sao as coordenadas do ponto I_j
     *     x e y sao as coordenadas da janela W
     */
    for (int x=ix-2; x<=jx+2; x++) {
        for (int y=iy-2; y<=jy+2; y++) {
//            std::cout << "Janela: x = " << x << " ix = " << ix << " jx = " << jx << "   y = " << y << " iy = " << iy << " jy = " << jy << std::endl;
//            std::cout << "Testando ponto " << x << "," << y << std::endl;
            if ( abs(x-ix)<=2 && abs(y-iy)<=2 && abs(x-jx)<=2 && abs(y-jy)<=2   && 
                 x > 0 && y > 0 && x < size[0] && y < size[1] 
                    ) {
//                std::cout << "Janela em  " << x << "," << y << "  contem os dois pontos" << std::endl;

                arma::mat tmp = I_i - Wmean.at(x+y*size[0]);
//                                tmp.print(std::cout, "TMP");
                arma::mat tmp2 = Wcov.at(x+y*size[0]) + (epsilon/(double)wk)*arma::eye(3,3);
//                                tmp2.print(std::cout, "TMP2");
                arma::mat tmp3 = I_j-Wmean.at(x+y*size[0]);
//                                tmp3.print(std::cout, "TMP3");
                arma::mat ltmp = tmp.t() * arma::inv(tmp2) * tmp3;
//                                ltmp.print(std::cout, "Ltmp");
                retorno += ( (ix==jx && iy==jy ? 1.0 : 0.0) - (1.0/(double) wk)*(1+ltmp(0,0)));

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



