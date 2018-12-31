/* This file is part of the Palabos library.
 *
 * Copyright (C) 2011-2017 FlowKit Sarl
 * Route d'Oron 2
 * 1010 Lausanne, Switzerland
 * E-mail contact: contact@flowkit.com
 *
 * The most recent release of Palabos can be downloaded at 
 * <http://www.palabos.org/>
 *
 * The library Palabos is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * The library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/** \file
 * Implementation of a stationary, pressure-driven 2D channel flow, and
 * comparison with the analytical Poiseuille profile. The velocity is initialized
 * to zero, and converges only slowly to the expected parabola. This application
 * illustrates a full production cycle in a CFD application, ranging from
 * the creation of a geometry and definition of boundary conditions over the
 * program execution to the evaluation of results and production of instantaneous
 * graphical snapshots. From a technical standpoint, this showcase is not
 * trivial: it implements for example hypbrid velocity/pressure boundaries, 
 * and uses an analytical profile to set up the boundary and initial conditions,
 * and to compute the error. As a first Palabos example, you might prefer to 
 * look at a more straightforward code, such as cavity2d.
 **/
 
#include "palabos2D.h"
#include "palabos2D.hh"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace plb;
using namespace plb::descriptors;
using namespace std;

typedef double T;
//#define DESCRIPTOR D2Q9Descriptor
#define DESCRIPTOR MRTD2Q9Descriptor

///// Velocity on the parabolic Poiseuille profile
//T poiseuilleVelocity(plint iY, IncomprFlowParam<T> const& parameters) {
//    T y = (T)iY / parameters.getResolution();
//    return 4.*parameters.getLatticeU() * (y-y*y);
//}
//
///// Linearly decreasing pressure profile
//T poiseuillePressure(plint iX, IncomprFlowParam<T> const& parameters) {
//    T Lx = parameters.getNx()-1;
//    T Ly = parameters.getNy()-1;
//    return 8.*parameters.getLatticeNu()*parameters.getLatticeU() / (Ly*Ly) * (Lx/(T)2-(T)iX);
//}
//
///// Convert pressure to density according to ideal gas law
//T poiseuilleDensity(plint iX, IncomprFlowParam<T> const& parameters) {
//    return poiseuillePressure(iX,parameters)*DESCRIPTOR<T>::invCs2 + (T)1;
//}
//
///// A functional, used to initialize the velocity for the boundary conditions
//template<typename T>
//class PoiseuilleVelocity {
//public:
//    PoiseuilleVelocity(IncomprFlowParam<T> parameters_)
//        : parameters(parameters_)
//    { }
//    void operator()(plint iX, plint iY, Array<T,2>& u) const {
//        u[0] = poiseuilleVelocity(iY, parameters);
//        u[1] = T();
//    }
//private:
//    IncomprFlowParam<T> parameters;
//};
//
///// A functional, used to initialize the density for the boundary conditions
//template<typename T>
//class PoiseuilleDensity {
//public:
//    PoiseuilleDensity(IncomprFlowParam<T> parameters_)
//        : parameters(parameters_)
//    { }
//    T operator()(plint iX, plint iY) const {
//        return poiseuilleDensity(iX,parameters);
//    }
//private:
//    IncomprFlowParam<T> parameters;
//};
//
///// A functional, used to create an initial condition for with zero velocity,
/////   and linearly decreasing pressure.
//template<typename T>
//class PoiseuilleDensityAndZeroVelocity {
//public:
//    PoiseuilleDensityAndZeroVelocity(IncomprFlowParam<T> parameters_)
//        : parameters(parameters_)
//    { }
//    void operator()(plint iX, plint iY, T& rho, Array<T,2>& u) const {
//        rho = poiseuilleDensity(iX,parameters);
//        u[0] = T();
//        u[1] = T();
//    }
//private:
//    IncomprFlowParam<T> parameters;
//};
//
//enum InletOutletT {pressure, velocity};
//
//void channelSetup( MultiBlockLattice2D<T,DESCRIPTOR>& lattice,
//                   IncomprFlowParam<T> const& parameters,
//                   OnLatticeBoundaryCondition2D<T,DESCRIPTOR>& boundaryCondition,
//                   InletOutletT inletOutlet )
//{
//    const plint nx = parameters.getNx();
//    const plint ny = parameters.getNy();
//
//    // Note: The following approach illustrated here works only with boun-
//    //   daries which are located on the outmost cells of the lattice. For
//    //   boundaries inside the lattice, you need to use the version of
//    //   "setVelocityConditionOnBlockBoundaries" which takes two Box2D
//    //   arguments.
//
//    // Velocity boundary condition on bottom wall. 
//    boundaryCondition.setVelocityConditionOnBlockBoundaries (
//            lattice, Box2D(0, nx-1, 0, 0) );
//    // Velocity boundary condition on top wall. 
//    boundaryCondition.setVelocityConditionOnBlockBoundaries (
//            lattice, Box2D(0, nx-1, ny-1, ny-1) );
//
//    // Pressure resp. velocity boundary condition on the inlet and outlet.
//    if (inletOutlet == pressure) {
//        // Note: pressure boundary conditions are currently implemented
//        //   only for edges of the boundary, and not for corner nodes.
//        boundaryCondition.setPressureConditionOnBlockBoundaries (
//                lattice, Box2D(0,0, 1,ny-2) );
//        boundaryCondition.setPressureConditionOnBlockBoundaries (
//                lattice, Box2D(nx-1,nx-1, 1,ny-2) );
//    }
//    else {
//        boundaryCondition.setVelocityConditionOnBlockBoundaries (
//                lattice, Box2D(0,0, 1,ny-2) );
//        boundaryCondition.setVelocityConditionOnBlockBoundaries (
//                lattice, Box2D(nx-1,nx-1, 1,ny-2) );
//    }
//
//    // Define the value of the imposed density on all nodes which have previously been
//    //   defined to be pressure boundary nodes.
//    setBoundaryDensity (
//            lattice, lattice.getBoundingBox(),
//            PoiseuilleDensity<T>(parameters) );
//    // Define the value of the imposed velocity on all nodes which have previously been
//    //   defined to be velocity boundary nodes.
//    setBoundaryVelocity (
//            lattice, lattice.getBoundingBox(),
//            PoiseuilleVelocity<T>(parameters) );
//    // Initialize all cells at an equilibrium distribution, with a velocity and density
//    //   value of the analytical Poiseuille solution.
//    initializeAtEquilibrium (
//           lattice, lattice.getBoundingBox(),
//           PoiseuilleDensityAndZeroVelocity<T>(parameters) );
//
//    // Call initialize to get the lattice ready for the simulation.
//    lattice.initialize();
//}

/// Produce a GIF snapshot of the velocity-norm.
void writeGif(MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint L, plint k, plint S, plint iter)
{
    plint nx = lattice.getNx() / 3;
    plint ny = lattice.getNy();
    Box2D pm(nx, 2*nx-1, 0, ny-1);
        
    const plint imSize = 600;

    string prefix = "u_"+std::to_string(L) + "_" + std::to_string(k) + "_" + std::to_string(S); 
    ImageWriter<T> imageWriter("leeloo");
    imageWriter.writeScaledGif(createFileName(prefix, iter, 6),
                               *computeVelocityNorm(lattice, pm),
                               imSize, imSize );
}

/// Write the full velocity and the velocity-norm into a VTK file.
void writeVTK(MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint L, plint k, plint S, plint iter)
{
//    T dx = parameters.getDeltaX();
//    T dt = parameters.getDeltaT();
    string prefix = "vtk_"+std::to_string(L) + "_" + std::to_string(k) + "_" + std::to_string(S); 
    VtkImageOutput2D<T> vtkOut(createFileName(prefix, iter, 6), 1.);
    vtkOut.writeData<float>(*computeVelocityNorm(lattice), "velocityNorm", 1.);
    vtkOut.writeData<2,float>(*computeVelocity(lattice), "velocity", 1.);
}

//T computeRMSerror ( MultiBlockLattice2D<T,DESCRIPTOR>& lattice,
//                    IncomprFlowParam<T> const& parameters )
//{
//    MultiTensorField2D<T,2> analyticalVelocity(lattice);
//    setToFunction( analyticalVelocity, analyticalVelocity.getBoundingBox(),
//                   PoiseuilleVelocity<T>(parameters) );
//    MultiTensorField2D<T,2> numericalVelocity(lattice);
//    computeVelocity(lattice, numericalVelocity, lattice.getBoundingBox());
//
//           // Divide by lattice velocity to normalize the error
//    return 1./parameters.getLatticeU() *
//           // Compute RMS difference between analytical and numerical solution
//               std::sqrt( computeAverage( *computeNormSqr(
//                              *subtract(analyticalVelocity, numericalVelocity)
//                         ) ) );
//}

class PressureGradient 
{
    public:
        PressureGradient(T deltaP_, plint n_ax_) : deltaP(deltaP_), n_ax(n_ax_) {}

        void operator() (plint iX, plint iY, T& rho, Array<T,2>& u) const
        {
            u.resetToZero();
            rho = 1. - deltaP * DESCRIPTOR<T>::invCs2 / (T)(n_ax - 1) * (T)iX;
        }

    private:
        T deltaP;
        plint n_ax;
};

Box2D setInlet(plint nx_, plint ny_)
{
    Box2D inlet (0, 0, 0, ny_-1);
    return inlet;
}

Box2D setOutlet(plint nx_, plint ny_)
{
    Box2D outlet (nx_-1, nx_-1, 0, ny_-1);
    return outlet;  
}

void porousMediaSetup( MultiBlockLattice2D<T,DESCRIPTOR>& lattice,
                       OnLatticeBoundaryCondition2D<T,DESCRIPTOR>* boundaryCondition,
                       MultiScalarField2D<int>& geometry, T deltaP, plint nx, plint ny)
{


        Box2D inlet = setInlet(nx, ny);
        boundaryCondition->addPressureBoundary0N(inlet, lattice);
        setBoundaryDensity(lattice, inlet, (T) 1.);

        Box2D outlet = setOutlet(nx, ny);
        boundaryCondition->addPressureBoundary0P(outlet, lattice);
        setBoundaryDensity(lattice, outlet, (T) 1. - deltaP*DESCRIPTOR<T>::invCs2);

        //pcerr << "Definition of the geometry." << endl;
        // Where "geometry" evaluates to 1, use bounce-back.
        defineDynamics(lattice, geometry, new BounceBack<T,DESCRIPTOR>(), 1);
        // Where "geometry" evaluates to 2, use no-dynamics (which does nothing).
        //defineDynamics(lattice, geometry, new NoDynamics<T,DESCRIPTOR>(), 1);

        pcerr << "Initilization of rho and u." << endl;
        initializeAtEquilibrium( lattice, lattice.getBoundingBox(), PressureGradient(deltaP, nx) );

        lattice.initialize();
        delete boundaryCondition;
}

T computeTortuosity(MultiBlockLattice2D<T,DESCRIPTOR> &lattice )
{

    plint nx = lattice.getNx() / 3;
    plint ny = lattice.getNy();
    Box2D pm(nx, 2*nx-1, 0, ny-1);

    T absvelsum = computeSum(*computeVelocityNorm(lattice, pm));
    T ax_velsum = computeSum(*computeVelocityComponent(lattice, pm, 0));

    T t = absvelsum / ax_velsum;
    return t;
}


T computePermeability ( MultiBlockLattice2D<T,DESCRIPTOR>& lattice, T nu, T deltaP, plint L, plint k, plint S)
{
        plint nx = lattice.getNx() / 3;
        plint ny = lattice.getNy();
        Box2D pm(nx, 2*nx-1, 0, ny-1);
        
        T meanU = computeAverage ( *computeVelocityComponent (lattice, pm, 0 ) );
        T t = computeTortuosity(lattice); 
   
        //pcout << meanU                           << " "; // average velocity
        //pcout << nu                              << " "; // Lattice viscosity nu
        //pcout << deltaP/(T)(n_ax-1)              << " "; // Grad P
        //pcout << nu*meanU / (deltaP/(T)(3*nx-1)) << " "; // Permeability
        //pcout << t                               << " \n"; // Tortuosity

        string prefix = std::to_string(L) + "_" + std::to_string(k) + "_" + std::to_string(S); 
        plb_ofstream ofile( ( global::directories().getVtkOutDir() + prefix + ".dat").c_str()  );
        ofile << nu*meanU / (deltaP/(T)(3*nx-1)) << " " << t << "\n";
        
        return meanU;
}

int main(int argc, char* argv[]) {
    plbInit(&argc, &argv);

    global::directories().setOutputDir("./tmp/");

//    IncomprFlowParam<T> parameters(
//            (T) 2e-2,  // uMax
//            (T) 5.,    // Re
//            60,        // N
//            3.,        // lx
//            1.         // ly 
//    );
    
    
    const plint nx_new    = atoi(argv[1]);
    const plint ny_new    = atoi(argv[2]);
    std::string fNameIn   = argv[3];
    const plint L    = atoi(argv[4]);
    const plint k    = atoi(argv[5]);
    const plint S    = atoi(argv[6]);
    
    const T omega = 1.0;
    const T deltaP    = 0.01;
    const T nu    = ((T)1/omega-0.5)/DESCRIPTOR<T>::invCs2;
    
    MultiBlockLattice2D<T,DESCRIPTOR> lattice_new(nx_new, ny_new, new MRTdynamics<T,DESCRIPTOR> ( omega ) );
    
    MultiScalarField2D<int> geometry(nx_new,ny_new);
    
    
    plb_ifstream geometryFile(fNameIn.c_str());
    if (!geometryFile.is_open()) 
    {
        pcerr << "Error: could not open geometry file " << fNameIn << endl;
        return -1;
    }
    geometryFile >> geometry;
    
    lattice_new.periodicity().toggle(0, false);
    lattice_new.periodicity().toggle(1, true);
    
    porousMediaSetup(lattice_new, createLocalBoundaryCondition2D<T,DESCRIPTOR>(), geometry, deltaP, nx_new, ny_new);
    
    util::ValueTracer<T> converge(1.0, 1000.0, 1.0e-5);

    //pcerr << "Simulation begins" << endl;
    plint iT = 0;

    plint maxT_ = 500000;
    
    while(true)
    {
//        if (iT % 5000 ==0) {
//            pcout << "Saving Gif ..." << endl;
//            writeGif(lattice_new, iT);
//            computePermeability (lattice_new, nu, deltaP);
//        }
        lattice_new.collideAndStream();
        converge.takeValue(getStoredAverageEnergy(lattice_new), true);
        
        if (converge.hasConverged())
        {
            writeGif(lattice_new, L, k, S, iT);
            writeVTK(lattice_new, L, k, S, iT);
            computePermeability (lattice_new, nu, deltaP, L, k, S);
            break;
        }

        if (iT >= maxT_)
        {
            break;
        }
        iT++;
        }
    
//    exit(1);
//    //******!!!(((((((!(((((((
//    
//    
//    const T logT     = (T)0.1;
//    const T imSave   = (T)0.5;
//    const T vtkSave  = (T)2.;
//    const T maxT     = (T)15.1;
//    // Change this variable to "pressure" if you prefer a pressure boundary
//    //   condition with Poiseuille profile for the inlet and the outlet.
//    const InletOutletT inletOutlet = pressure;
//
//    writeLogFile(parameters, "Poiseuille flow");
//
//    MultiBlockLattice2D<T, DESCRIPTOR> lattice (
//              parameters.getNx(), parameters.getNy(),
//              new BGKdynamics<T,DESCRIPTOR>(parameters.getOmega()) );
//
//    OnLatticeBoundaryCondition2D<T,DESCRIPTOR>*
//        boundaryCondition = createLocalBoundaryCondition2D<T,DESCRIPTOR>();
//
//    channelSetup(lattice, parameters, *boundaryCondition, inletOutlet);
//
//    // Main loop over time iterations.
//    for (plint iT=0; iT*parameters.getDeltaT()<maxT; ++iT) {
//       if (iT%parameters.nStep(imSave)==0) {
//            pcout << "Saving Gif ..." << endl;
//            writeGif(lattice, iT);
//        }
//
//        if (iT%parameters.nStep(vtkSave)==0 && iT>0) {
//            pcout << "Saving VTK file ..." << endl;
//            writeVTK(lattice, parameters, iT);
//        }
//
//        if (iT%parameters.nStep(logT)==0) {
//            pcout << "step " << iT
//                  << "; t=" << iT*parameters.getDeltaT()
//                  << "; RMS error=" << computeRMSerror(lattice, parameters);
//            Array<T,2> uCenter;
//            lattice.get(parameters.getNx()/2,parameters.getNy()/2).computeVelocity(uCenter);
//            pcout << "; center velocity=" << uCenter[0]/parameters.getLatticeU() << endl;
//        }
//
//        // Lattice Boltzmann iteration step.
//        lattice.collideAndStream();
//    }
//
//    delete boundaryCondition;
}
