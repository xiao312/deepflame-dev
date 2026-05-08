#include "O2.H"
#include "addToRunTimeSelectionTable.H"


namespace Foam
{
    defineTypeNameAndDebug(O2, 0);
    addToRunTimeSelectionTable(liquidProperties, O2,);
    addToRunTimeSelectionTable(liquidProperties, O2, dictionary);
}


Foam::O2::O2()
:
    liquidProperties
    (
        0.031999,
        154.5994,
        5.0464e+6,
        0.07495,
        0.294248,
        54.361,
        146.28,
        90.188,
        -1,
        0.0222,
        14710.24
    ),

    pv_(2.68794635e+01, -9.58794777e+02, -1.05052914e+00, -6.90048355e-05, 1.21846002e-02),

    rho_(1.14632554e+02, 2.75655615e-01, 154.5994, 2.77505361e-01),

    hl_(154.5994, 2.89780460e+05, 6.00000000e-01, -7.05482356e-01, 4.75635399e-01, 0.0),

    Cp_(154.5994, 1.09590080e+01, 1.45625217e+03, 2.33843393e+01, -8.99949070e+01),

    h_(-9.68908452e+05, 2.50637084e+04, -5.16097446e+02, 5.55884345e+00, -2.93205446e-02, 6.09448175e-05),

    mu_(-2.11200494e+00, 7.62848791e+01, -1.61677107e+00, -7.70004913e-30, 1.31855757e+01),

    sigma_(154.5994, 3.83537868e-02, 1.21069008e+00, 4.43081971e-02, -5.45942334e-02, 0.0),

    kappa_(-9.19925611e-02, 1.93351101e-02, -4.51801714e-04, 4.79054225e-06, -2.47777620e-08, 5.00382658e-11),

    Cpg_(154.5994, 1.36975300e+01, 3.52746624e+02, -6.80059012e+00, -4.86846476e+01),

    B_(1.20601567e-02, -6.15471876e-04, 1.22024524e-05, -1.17761704e-07, 5.54479099e-10, -1.02125637e-12),

    mug_(-1.76036598e+01, 1.50014787e+01, 1.23492678e+00, 1.17825779e-56, 2.53862523e+01),

    kappag_(-9.52797778e-01, 5.36816017e-02, -1.17434164e-03, 1.25352582e-05, -6.52648492e-08, 1.32897143e-10),

    D_(16.3, 18.5, 31.999, 28.0)
{}


Foam::O2::O2
(
    const liquidProperties& l,
    const NSRDSfunc5& density, 
    const NSRDSfunc1& vapourPressure,
    const NSRDSfunc6& heatOfVapourisation, 
    const NSRDSfunc14& heatCapacity,
    const NSRDSfunc0& enthalpy,
    const NSRDSfunc14& idealGasHeatCapacity,
    const NSRDSfunc0& secondVirialCoeff,
    const NSRDSfunc1& dynamicViscosity,
    const NSRDSfunc1& vapourDynamicViscosity, 
    const NSRDSfunc0& thermalConductivity,
    const NSRDSfunc0& vapourThermalConductivity,
    const NSRDSfunc6& surfaceTension,
    const APIdiffCoefFunc& vapourDiffussivity
)
:
    liquidProperties(l),
    rho_(density),
    pv_(vapourPressure),
    hl_(heatOfVapourisation),
    Cp_(heatCapacity),
    h_(enthalpy),
    Cpg_(idealGasHeatCapacity),
    B_(secondVirialCoeff),
    mu_(dynamicViscosity),
    mug_(vapourDynamicViscosity),
    kappa_(thermalConductivity),
    kappag_(vapourThermalConductivity),
    sigma_(surfaceTension),
    D_(vapourDiffussivity)
{}


Foam::O2::O2(const dictionary& dict)
:
    O2()
{
    readIfPresent(*this, dict);
}



void Foam::O2::writeData(Ostream& os) const
{
    liquidProperties::writeData(*this, os);
}



Foam::Ostream& Foam::operator<<(Ostream& os, const O2& l)
{
    l.writeData(os);
    return os;
}