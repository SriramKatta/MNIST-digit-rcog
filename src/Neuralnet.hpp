#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cfloat>

#include "NueralNetparams.hpp"
#include "mnistdatareader.hpp"

constexpr size_t lay = 2;
/*
inline void matrixprinter(const Eigen::MatrixXd &mat)
{
  Eigen::IOFormat CleanFmt(2, 0, ",", "\n");
  std::cout << mat.format(CleanFmt) << std::endl;
}
*/

class Neuralnet
{
  using eigenVector = Eigen::VectorXd;
  using eigenMatrix = Eigen::MatrixXd;
  using eigenBlock = Eigen::Block<eigenMatrix>;

public: // functions
  Neuralnet(const size_t& inputsize, const size_t& no_of_outputs, const NeuralNetworkParams &nnparm) : 
            inputlayersize(inputsize), 
            output_classes(no_of_outputs), 
            nnparams(nnparm), 
            intrainmode(true), 
            output()
  {
    weight_bias[0] = eigenMatrix::Random(nnparams.hidden_size, inputlayersize + 1); // 500 x 785
    weight_bias[1] = eigenMatrix::Random(output_classes, nnparams.hidden_size + 1); // 10 x 501
  }

  void train()
  {
    intrainmode = true;
  }

  void eval()
  {
    intrainmode = false;
  }

  eigenMatrix operator()(const eigenMatrix &img, const eigenMatrix &labels = {})
  {
    forwardPass(img);
    if (intrainmode)
    {
      backwardPass(img, labels);
      return crossentropyloss(labels);
    }
    return output;
  }

  inline eigenMatrix crossentropyloss(const eigenMatrix &actual_out) const
  {
    return -(actual_out.array() * output.array().log()).colwise().sum();
  }

private: // functions
  inline void forwardPass(const eigenMatrix &images)
  {
    hiddernlayers = weight_bias[0] * appended(images);              // 500x785 * 784+1xB = 500xB
    activationlayers = ReLU(hiddernlayers);                      // 500xB
    output = softmax(weight_bias[1] * appended(activationlayers));                                      // 10xB
  }

  inline void backwardPass(const eigenMatrix &images, const eigenMatrix &actual_out)
  {
    eigenMatrix dz1 = output - actual_out;                                        // 10 x B
    eigenMatrix dw = dz1 * activationlayers.transpose() / nnparams.batch_size; // 10x500
    eigenMatrix db = dz1.rowwise().sum() / nnparams.batch_size;                   // 10 x 1
    eigenMatrix dwb1(weight_bias[1].rows(), weight_bias[1].cols());               // 10 x 501 
    dwb1 << dw,db;
    eigenMatrix dz0 = (weight_bias[1].leftCols(weight_bias[1].cols()-1).transpose() * dz1).array() 
                                    * dev_ReLU(hiddernlayers).array();         //500 x 10 * 10 x B .* 500 x B = 500 x B
    dw = dz0 * images.transpose() ;                                               // 500xB * Bx784 = 500x784
    db = dz0.rowwise().sum() / nnparams.batch_size;                               // 500 x 1

    eigenMatrix dwb0(weight_bias[0].rows(), weight_bias[0].cols());               // 500x785
    dwb0 << dw,db;
   
    SGD_update(dwb0, dwb1);
  }

  inline void parallel_col_fill(eigenMatrix &mat, const eigenVector &vec)
  {
    mat = vec.replicate(1, mat.cols());
  }

  inline eigenMatrix dev_ReLU(const eigenMatrix &mat)
  {
    return (mat.array() >= 0).cast<double>();
  }

  inline eigenMatrix ReLU(const eigenMatrix &hlayer)
  {
    return hlayer.cwiseMax(0);
  }

  inline eigenMatrix softmax(eigenMatrix alayer)
  {
    alayer.rowwise() -= alayer.colwise().maxCoeff();
    alayer = alayer.array().exp();
    eigenVector val = alayer.colwise().sum();
    for (size_t i = 0; auto coloumref : alayer.colwise())
      coloumref /= val(i++);
    return alayer;
  }

  inline void SGD_update(const eigenMatrix &dwb0, const eigenMatrix &dwb1)
  {
    weight_bias[0] -= nnparams.learning_rate * dwb0;
    weight_bias[1] -= nnparams.learning_rate * dwb1;
  }

  eigenMatrix appended(const eigenMatrix &blk)
  {
    eigenMatrix newmat(blk.rows() + 1, blk.cols());
    newmat << blk, Eigen::RowVectorXd::Ones(blk.cols());
    return newmat;
  }

private: // datamembers
  const size_t inputlayersize;
  const size_t output_classes;
  const NeuralNetworkParams nnparams;
  std::array<eigenMatrix, lay> weight_bias;
  eigenMatrix hiddernlayers;
  eigenMatrix activationlayers;
  eigenMatrix output;
  bool intrainmode;
};

/*working version for seperate weight and bias

    eigenMatrix dz1 = output - actual_out;                                         // 10 x 100
    eigenMatrix dw1 = dz1 * activationlayers[0].transpose() / nnparams.batch_size; // 500x10
    eigenVector dbfill = dz1.rowwise().sum() / nnparams.batch_size;                // 10 x 1
    eigenMatrix db1 = dbfill.replicate(1, bias[1].cols());                         // 10 x 100
    eigenMatrix dz0 = (weight_bias[1].transpose() * dz1).array() * dev_ReLU(hiddernlayers[0]).array();
    //                  500 x 10 * 10 x 100 .* 500 x 100 = 500 x 100
    eigenMatrix dw0 = dz0 * images.transpose();            // 784x100 * 100 x 500
    dbfill = dz0.rowwise().sum() / nnparams.batch_size;    // 500 x 1
    eigenMatrix db0 = dbfill.replicate(1, bias[0].cols()); // 500 x b

*/

/*
    eigenMatrix en = -actual_out.array()*  (output.array().inverse());
    matrixprinter(output.array());
    matrixprinter(actual_out.array());
    matrixprinter(en.array());
    eigenMatrix temp = (en - activationlayers[1]).colwise().sum().replicate(10,1);
    eigenMatrix en1 = output - (en - temp); //10x100
    eigenMatrix dw1 = en1 * activationlayers[0].transpose();

    ////assuming true
    eigenMatrix db1 = 1/nnparams.batch_size * en1.rowwise().sum().replicate(1,nnparams.batch_size);

    eigenMatrix en2 = (weights[1].transpose() *  en1).array() * dev_ReLU(hiddernlayers[0]).array();

    eigenMatrix dw0 = en2 * images.transpose();

    eigenMatrix db0 = 1/nnparams.batch_size * en2.rowwise().sum().replicate(1,nnparams.batch_size);
*/

/*
    eigenMatrix en = -(actual_out.array() / output.array()); // 10 x 100
    eigenMatrix ent = en.transpose(); // 100 x 10
    eigenMatrix en_1 = output.array() - (en.array() - en.sum() - output.array()); // 10 x 100

    eigenMatrix endoten_1t = en.array() * en_1.array();
    endoten_1t.transposeInPlace();

    eigenMatrix dw1 = activationlayers[0]  * endoten_1t; // 500 x 100 . 100 x 10 . 10 x  100 = 500 x100

    eigenVector dbfill1 = (en.array() * en_1.array()).rowwise().sum()/en.rows(); // 10 x 1
    eigenMatrix db1(bias[1].rows(), bias[1].cols());
    parallel_col_fill(db1,dbfill1);

    eigenMatrix devrel = dev_ReLU(weights.back());

    eigenMatrix dw0 = images * endoten_1t * (devrel.transpose());
    eigenVector dbfill0 =  devrel * dbfill1;
    eigenMatrix db0(bias[0].rows(), bias[0].cols());
    parallel_col_fill(db0,dbfill0);

*/

/*
eigenMatrix en = actual_out.array() / output.array();   // 10 x 100
    eigenMatrix en_1 = output - actual_out; // 10 x 100

    eigenMatrix enn_1 =  en.transpose() * en_1; //100x100

    eigenMatrix dw1 = (activationlayers[0] * enn_1)/nnparams.batch_size; // (500X100)' * 10 X 100

    eigenVector dbfill = (enn_1.rowwise().sum() / enn_1.rows())/nnparams.batch_size;
    eigenMatrix db1 = dbfill.replicate(1, bias[1].cols());

    eigenMatrix enrelu = dev_ReLU(hiddernlayers[1]) * en_1.transpose() * en; // 500x100
    eigenMatrix enw = weights[1] * enrelu;
    eigenMatrix dw0 = (images * enw.transpose())/nnparams.batch_size;

    dbfill = enw.rowwise().sum();
    eigenMatrix db0 = dbfill.replicate(1, bias[0].cols());

*/

 
    /* producing NAN
    eigenMatrix en = -actual_out.array() * (output.array() +  DBL_EPSILON).inverse();  //10xB
    if(en.hasNaN()){
      std::cout << "has nan" << std::endl;
    }
    eigenMatrix en_1 = hiddernlayers[1] - en + (en - hiddernlayers[1]).colwise().sum().replicate(en.rows(),1);
    //10XB
    eigenMatrix dw = en_1 * activationlayers[0].transpose(); //10XB * BX500 = 10 x 500
    eigenMatrix db = en_1.rowwise().sum() / nnparams.batch_size; //10x1

    eigenMatrix dwb1(weight_bias[1].rows(),weight_bias[1].cols());
    dwb1 << dw,db;
    eigenMatrix blk = weight_bias[1].leftCols(weight_bias[1].cols() - 1).transpose();
    eigenMatrix en_2 = (blk * en_1).array() * dev_ReLU(hiddernlayers[0]).array(); //10xB * 500xB

    dw = en_2 * images.transpose(); //   500xB * Bx784 = 500x784
    db = en_2.rowwise().sum() / nnparams.batch_size;

    eigenMatrix dwb0(weight_bias[0].rows(),weight_bias[0].cols()); //500x785
    dwb0 << dw,db;
    */
