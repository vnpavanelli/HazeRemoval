#include "itkImage.h"
#include "itkImageFileReader.h"
#include "QuickView.h"
 
template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);
 
int main(int argc, char *argv[])
{
  if(argc < 2)
    {
    std::cerr << "Required: filename" << std::endl;
 
    return EXIT_FAILURE;
    }
  std::string inputFilename = argv[1];
 
  typedef itk::ImageIOBase::IOComponentType ScalarPixelType;
  typedef itk::RGBPixel< unsigned char >    PixelType;
 
  itk::ImageIOBase::Pointer imageIO =
        itk::ImageIOFactory::CreateImageIO(
            inputFilename.c_str(), itk::ImageIOFactory::ReadMode);
  if( !imageIO )
  {
    std::cerr << "Could not CreateImageIO for: " << inputFilename << std::endl;
    return EXIT_FAILURE;
  }
  imageIO->SetFileName(inputFilename);
  imageIO->ReadImageInformation();
  const ScalarPixelType pixelType = imageIO->GetComponentType();
  std::cout << "Pixel Type is " << imageIO->GetComponentTypeAsString(pixelType) // 'double'
            << std::endl;
  const size_t numDimensions =  imageIO->GetNumberOfDimensions();
  std::cout << "numDimensions: " << numDimensions << std::endl; // '2'
 
  std::cout << "component size: " << imageIO->GetComponentSize() << std::endl; // '8'
  std::cout << "pixel type (string): " << imageIO->GetPixelTypeAsString(imageIO->GetPixelType()) << std::endl; // 'vector'
  std::cout << "pixel type: " << imageIO->GetPixelType() << std::endl; // '5'

  typedef itk::Image<PixelType, 2> ImageType;
  typedef itk::Image<unsigned char, 2> ImageGrayType;
  ImageType::Pointer image = ImageType::New();
  ImageType::Pointer image2 = ImageType::New();
  ImageGrayType::Pointer image_min = ImageGrayType::New();
  ImageGrayType::Pointer image_dark = ImageGrayType::New();
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

  image2->SetRegions(region);
  image2->Allocate();

  std::cout << "Criando imagem minima..." << std::endl;
  /* Criar imagem com minimo dentre os canais RGB */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          auto pixel = image->GetPixel({i,j});
          unsigned char min = std::min(pixel[0], std::min(pixel[1], pixel[2]));
          image_min->SetPixel({i, j}, min);
      }
  }

  const unsigned int Patch_window = 15;

  std::cout << "Criando Dark Channel..." << std::endl;
  /* Criar o Prior Dark Channel usando o minimo de um filtro Patch_window X Patch_window (e.g. 15x15) */
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          unsigned char min = 255;
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

  std::cout << "Procurando valor de A..." << std::endl;
  unsigned int total_pixels = size[0]*size[1];
  int porcento01 = ceil((double) total_pixels * 0.001);
  unsigned char pixel_max = 255;
  std::cout << "Total pixels: " << total_pixels << std::endl;
  std::cout << "0.1%: " << porcento01 << std::endl;
  PixelType::LuminanceType pixel_A=0;
  do {
      for (unsigned int i = 0; i < size[0] && porcento01 > 0; i++) {
          for (unsigned int j = 0; j < size[1] && porcento01 > 0; j++) {
              unsigned char pixel = image_dark->GetPixel({i,j});
              if (pixel >= pixel_max) {
                  /*
                  std::cout << " -> pixel: " << pixel << " >= " << pixel_max << "   porcento01:" << porcento01 << std::endl;
                  std::cout << "    Luminance: " << image->GetPixel({i,j}).GetLuminance() << std::endl;
                  */
                  porcento01--;
                  pixel_A = std::max(pixel_A, image->GetPixel({i,j}).GetLuminance());
              }
          }
      }
      pixel_max--;
  } while (porcento01>0 && pixel_max >0);

  std::cout << "Valor de A atmosferico: " << pixel_A << std::endl;

//  double A = (double) pixel_A/255;

  unsigned char A = pixel_A;
  const unsigned char t_max = 25;
  for (unsigned int i = 0; i < size[0]; i++) {
      for (unsigned int j = 0; j < size[1]; j++) {
          PixelType pixel = image->GetPixel({i,j});
          PixelType pixelout;
          unsigned char t_dark = 255-(image_dark->GetPixel({i,j}));
          double t = (double) std::max(t_max,t_dark) / 255.0;
          pixelout[0] = (double) (pixel[0] - A)/t + A; 
          pixelout[1] = (double) (pixel[1] - A)/t + A; 
          pixelout[2] = (double) (pixel[2] - A)/t + A; 
          /*
          std::cout << " -> (" << i << "," << j << ") t: " << (int) t << " pixel0: " << (int) pixel[0] << " pixelout0: " << (int) pixelout[0] 
               << "P-A:" << (int) pixel[0]-A << " P-A/t: " <<(int)  (pixel[0]-A)/t
              << std::endl;
              */
          image2->SetPixel({i,j}, pixelout);
      }
  }



  QuickView viewer;
  viewer.AddImage(image.GetPointer());
  viewer.AddImage(image2.GetPointer());
//  viewer.AddImage(image_min.GetPointer());
//  viewer.AddImage(image_dark.GetPointer());
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
