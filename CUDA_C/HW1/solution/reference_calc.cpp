void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols)
{
  for (size_t r = 0; r < numRows; ++r) {
    for (size_t c = 0; c < numCols; ++c) {
      uchar4 rgba = rgbaImage[r * numCols + c];
      unsigned int channelSum = rgba.x + rgba.y + rgba.z;
      greyImage[r * numCols + c] = channelSum / 3;
    }
  }
}

