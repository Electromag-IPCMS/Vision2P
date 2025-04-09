"""
Vision2P - A data vision software package that enables advanced AI and machine learning based analysis. It combines clustering and unmixing through a modified constrained Non-Negative Matrix Factorization (NMF) algorithm with advanced features tailored for spectromicroscopy applications.

Copyright © 2025 CNRS and Université de Strasbourg

Authors: Boris Croes and Salia Cherifi-Hertel

This program is free software: you can redistribute it and/or modify it under the terms of the 3-Clause BSD License. You should have received a copy of the 3-Clause BSD License along with this program.  If not, see <https://opensource.org/license/BSD-3-Clause>.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Contacts: 
Boris Croes, IPCMS Strasbourg, 23 rue du Loess,67034 Strasbourg, France. boris.croes@ipcms.unistra.fr
Salia Cherifi-Hertel, IPCMS Strasbourg, 23 rue du Loess,67034 Strasbourg, France. salia.cherifi@ipcms.unistra.fr
"""

from .kmeans import kmeans
from .nmf import nmf