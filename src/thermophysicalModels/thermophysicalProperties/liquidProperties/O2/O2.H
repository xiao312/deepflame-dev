/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::O2

Description
    Liquid Oxygen

SourceFiles
    O2.C

\*---------------------------------------------------------------------------*/

#ifndef O2_H
#define O2_H

#include "liquidProperties.H"
#include "NSRDSfunc0.H"
#include "NSRDSfunc1.H"
#include "NSRDSfunc5.H"
#include "NSRDSfunc6.H"
#include "NSRDSfunc14.H"
#include "APIdiffCoefFunc.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


class O2
:
    public liquidProperties
{

        NSRDSfunc5 rho_;
        NSRDSfunc1 pv_;
        NSRDSfunc6 hl_;
        NSRDSfunc14 Cp_;
        NSRDSfunc0 h_;
        NSRDSfunc14 Cpg_;
        NSRDSfunc0 B_;
        NSRDSfunc1 mu_;
        NSRDSfunc1 mug_;
        NSRDSfunc0 kappa_;
        NSRDSfunc0 kappag_;
        NSRDSfunc6 sigma_;
        APIdiffCoefFunc D_;


public:

    friend class liquidProperties;

    TypeName("O2");



        O2();

        O2
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
        );

        O2(const dictionary& dict);

        virtual autoPtr<liquidProperties> clone() const
        {
            return autoPtr<liquidProperties>(new O2(*this));
        }


        inline scalar rho(scalar p, scalar T) const;

        inline scalar pv(scalar p, scalar T) const;

        inline scalar hl(scalar p, scalar T) const;

        inline scalar Cp(scalar p, scalar T) const;

        inline scalar h(scalar p, scalar T) const;

        inline scalar Cpg(scalar p, scalar T) const;

        inline scalar B(scalar p, scalar T) const;

        inline scalar mu(scalar p, scalar T) const;

        inline scalar mug(scalar p, scalar T) const;

        inline scalar kappa(scalar p, scalar T) const;

        inline scalar kappag(scalar p, scalar T) const;

        inline scalar sigma(scalar p, scalar T) const;

        inline scalar D(scalar p, scalar T) const;

        inline scalar D(scalar p, scalar T, scalar Wb) const;


        void writeData(Ostream& os) const;

        friend Ostream& operator<<(Ostream& os, const O2& l);
};

Ostream& operator<<(Ostream& os, const O2& l);

} 

#include "O2I.H"


#endif

// ************************************************************************* //