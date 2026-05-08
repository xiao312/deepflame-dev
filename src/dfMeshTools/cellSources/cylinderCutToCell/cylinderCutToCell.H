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
    Foam::cylinderCutToCell

Description
    A topoSetSource to select cells that intersect with a cylinder.
    A cell is selected if any of its vertices is inside the cylinder,
    or if the cylinder axis passes through the cell's bounding box.

    Example usage in topoSetDict:
    \verbatim
        {
            name    cylinderCut;
            type    cellSet;
            action  new;
            source  cylinderCutToCell;
            sourceInfo
            {
                p1      (0 0 0.015);
                p2      (0 0 0.025);
                radius  0.0275;
            }
        }
    \endverbatim

SourceFiles
    cylinderToCell.C

\*---------------------------------------------------------------------------*/

#ifndef cylinderCutToCell_H
#define cylinderCutToCell_H

#include "topoSetSource.H"
#include "boundBox.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class cylinderCutToCell Declaration
\*---------------------------------------------------------------------------*/

class cylinderCutToCell
:
    public topoSetSource
{

    // Private Data

        //- Add usage string
        static addToUsageTable usage_;

        //- First point on cylinder axis
        vector p1_;

        //- Second point on cylinder axis
        vector p2_;

        //- Radius
        scalar radius_;


    // Private Member Functions

        //- Check if a point is inside the cylinder
        bool pointInCylinder
        (
            const point& pt,
            const vector& axis,
            const scalar magAxis2,
            const scalar rad2
        ) const;

        //- Check if cylinder intersects with a bounding box
        bool cylinderIntersectsBoundBox
        (
            const boundBox& bb,
            const vector& axis,
            const vector& axisDir,
            const scalar magAxis,
            const scalar rad2
        ) const;

        void combine(topoSet& set, const bool add) const;


public:

    //- Runtime type information
    TypeName("cylinderCutToCell");


    // Constructors

        //- Construct from components
        cylinderCutToCell
        (
            const polyMesh& mesh,
            const vector& p1,
            const vector& p2,
            const scalar radius
        );

        //- Construct from dictionary
        cylinderCutToCell
        (
            const polyMesh& mesh,
            const dictionary& dict
        );

        //- Construct from Istream
        cylinderCutToCell
        (
            const polyMesh& mesh,
            Istream&
        );


    //- Destructor
    virtual ~cylinderCutToCell();


    // Member Functions

        virtual sourceType setType() const
        {
            return CELLSETSOURCE;
        }

        virtual void applyToSet
        (
            const topoSetSource::setAction action,
            topoSet&
        ) const;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
