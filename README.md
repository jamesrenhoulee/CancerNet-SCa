# CancerNet-SCa

**Note: The CancerNet-SCa models provided here are intended to be used as reference models that can be built upon and enhanced as new data becomes available. They are currently at a research stage and not yet intended as production-ready models (not meant for direct clinical diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use CancerNet-SCa for self-diagnosis and seek help from your local health authorities.**

**CancerNet-SCa is part of the CancerNet initiatives, a parallel initiative to the [COVID-Net initiative](https://github.com/lindawangg/COVID-Net).**

If you are a researcher or healthcare worker and you would like access to the **GSInquire tool to use to interpret CancerNet-SCa results** on your data or existing data, please reach out to a28wong@uwaterloo.ca or alex@darwinai.ca

Our desire is to encourage broad adoption and contribution to this project. Accordingly this project has been licensed under the GNU Affero General Public License 3.0. Please see [license file](LICENSE.md) for terms.

If there are any technical questions after the README, FAQ, and past/current issues have been read, please post an issue or contact:
* jamesrenhoulee@gmail.com

## Quick Links
1. Main ISIC Archive: https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main
2. CancerNet-SCa models (skin cancer detection via dermoscopy images): https://drive.google.com/drive/folders/1COrqcPMVqEf9qwnYY0HK_32Ka_3SWG8b?usp=sharing


## Core SkinCancerNet Team
* DarwinAI Corp., Canada and Vision and Image Processing Research Group, University of Waterloo, Canada
	* Alexander Wong
* Vision and Image Processing Research Group, University of Waterloo, Canada
	* James Lee
* DarwinAI Corp., Canada
	* Mahmoud Famouri
* University of Waterloo, Canada
	* Maya Pavlova

## Table of Contents
1. [Requirements](#requirements) to install, train, and infer CancerNet-SCa on your system.

## Requirements

The main requirements are listed below:

* Tensorflow 1.15
* OpenCV 4.2.0
* Python 3.6
* Numpy

Additional requirements to generate dataset:

* Pandas
* Jupyter

## Results

These are the final results for the CancerNet-SCa models.

### CancerNet-SCa-A on ISIC test set
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="2">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Benign</td>
    <td class="tg-7btt">Malignant</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="2">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Benign</td>
    <td class="tg-7btt">Malignant</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</table></div>

### CancerNet-SCa-B on ISIC test set
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="2">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Benign</td>
    <td class="tg-7btt">Malignant</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="2">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Benign</td>
    <td class="tg-7btt">Malignant</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</table></div>

### CancerNet-SCa-C on ISIC test set
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="2">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Benign</td>
    <td class="tg-7btt">Malignant</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="2">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Benign</td>
    <td class="tg-7btt">Malignant</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
</table></div>
