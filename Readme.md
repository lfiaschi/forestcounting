Learning to Count with a Regression Forest and Structured Labels
============================================

                Copyright 2012 by Luca Fiaschi
                
                luca.fiaschi@iwr.uni-heidelberg.com
                
                http://hci.iwr.uni-heidelberg.de/Staff/lfiaschi/


	This demo show how to count objects in images with a regression forest.
	
    If you use this code, please use the following citation in all relevant publications:
    @article{fiaschi_12_learning,	
  	 title={Learning to Count with Regression Forest and Structured Labels},
  	 author={Fiaschi, L. and Nair, R. and Koethe, U. and Hamprecht, F.A.},
  	 journal={ICPR 2012. Proceedings},
   	 year={2012}
	 }
    
    
    
    
    You may use, modify, and distribute this software according
    to the terms stated in the LICENSE.txt file.
                         
    THIS SOFTWARE IS PROVIDED AS IS AND WITHOUT ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
    WARRANTIES OF MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.


Installation
------------
This software has the following depend on:
numpy
cython
scikits-learn

At first you need to compiele the cython files:
on linux machines run 
./compile.sh

The demo is contained in the file demo.py

Optional dependency:
	matplotlib
