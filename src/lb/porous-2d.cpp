#include "palabos2D.h"
#include "palabos2D.hh"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

// THE FOLLOWING CODE IS PARTIALLY BASED ON:
// https://github.com/FlowKit/palabos-examples/blob/master/showCases/poiseuille/poiseuille.cpp

using namespace plb;
using namespace plb::descriptors;
using namespace std;

typedef double T;
//#define DESCRIPTOR D2Q9Descriptor
#define DESCRIPTOR MRTD2Q9Descriptor

/// Produce a GIF snapshot of the velocity-norm.
void writeGif(MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint L, plint k, plint S, plint iter)
{
    plint nx = lattice.getNx() / 3;
    plint ny = lattice.getNy();
    Box2D pm(nx, 2*nx-1, 0, ny-1);
        
    const plint imSize = 600;

    string prefix = "u_"+std::to_string(L) + "_" + std::to_string(k) + "_" + std::to_string(S) + "_"; 
    ImageWriter<T> imageWriter("leeloo");
    imageWriter.writeScaledGif(createFileName(prefix, iter, 6),
                               *computeVelocityNorm(lattice, pm),
                               imSize, imSize );
}

void saveFlowField(MultiBlockLattice2D<T,DESCRIPTOR>& lattice, plint L, plint k, plint S, plint iter)
{
    FileName fn = createFileName("ux_uy_"+std::to_string(L) + "_" + std::to_string(k) + "_" + std::to_string(S) + "_", iter, 6);
    plb_ofstream ofile( ( global::directories().getVtkOutDir() + fn.get() + ".dat").c_str()  );
   
    Box2D domain = lattice.getBoundingBox();
    std::auto_ptr< MultiScalarField2D< T > > UX = computeVelocityComponent (lattice, domain, 0 );
    std::auto_ptr< MultiScalarField2D< T > > UY = computeVelocityComponent (lattice, domain, 1 );
    std::auto_ptr< MultiScalarField2D< T > > U  = computeVelocityNorm (lattice, domain );

    T u, ux, uy;

    plint nx = lattice.getNx();
    plint ny = lattice.getNy();
    for (plint ix = 0; ix < nx; ix++)
    {
        for (plint iy = 0; iy < ny; iy++)
        {
            u  = U->get(ix, iy);
            ux = UX->get(ix, iy);
            uy = UY->get(ix, iy);
            ofile << ix << " " << iy << " " << ux << " " << uy << " " << u << endl;
        }
    }
}

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

        defineDynamics(lattice, geometry, new BounceBack<T,DESCRIPTOR>(), 1);
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
        plb_ofstream ofile( ( global::directories().getVtkOutDir() + "/" + prefix + ".dat").c_str()  );
        ofile << nu*meanU / (deltaP/(T)(3*nx-1)) << " " << t << "\n";
        
        return meanU;
}

int main(int argc, char* argv[]) {
    plbInit(&argc, &argv);

    const plint nx_new    = atoi(argv[1]);
    const plint ny_new    = atoi(argv[2]);
    std::string fNameIn   = argv[3];
    const plint L    = atoi(argv[4]);
    const plint k    = atoi(argv[5]);
    const plint S    = atoi(argv[6]);
    
    std::string opath = argv[7];
    global::directories().setOutputDir(opath);
    
    const T omega  = 1.0;
    const T deltaP = 0.01;
    const T nu     = ((T)1/omega-0.5)/DESCRIPTOR<T>::invCs2;
    
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

    plint iT = 0;
    plint maxT_ = 500000;
    
    while(true)
    {
        lattice_new.collideAndStream();
        converge.takeValue(getStoredAverageEnergy(lattice_new), true);
        
        if (converge.hasConverged())
        {
            writeGif(lattice_new, L, k, S, iT);
            saveFlowField(lattice_new, L, k, S, iT);
            computePermeability (lattice_new, nu, deltaP, L, k, S);
            break;
        }

        if (iT >= maxT_)
        {
            break;
        }
        iT++;
    }
}
