/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

#include "cylinderCutToCell.H"
#include "polyMesh.H"
#include "addToRunTimeSelectionTable.H"
#include "cellModel.H"
#include "boundBox.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(cylinderCutToCell, 0);
    addToRunTimeSelectionTable(topoSetSource, cylinderCutToCell, word);
    addToRunTimeSelectionTable(topoSetSource, cylinderCutToCell, istream);
}


Foam::topoSetSource::addToUsageTable Foam::cylinderCutToCell::usage_
(
    cylinderCutToCell::typeName,
    "\n    Usage: cylinderCutToCell (p1X p1Y p1Z) (p2X p2Y p2Z) radius\n\n"
    "    Select all cells that intersect with the bounding cylinder\n\n"
);


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

// Helper function: check if a point is inside the cylinder
bool Foam::cylinderCutToCell::pointInCylinder
(
    const point& pt,
    const vector& axis,
    const scalar magAxis2,
    const scalar rad2
) const
{
    vector d = pt - p1_;
    scalar magD = d & axis;

    if ((magD >= 0) && (magD <= magAxis2))
    {
        scalar d2 = (d & d) - sqr(magD)/magAxis2;
        if (d2 <= rad2)
        {
            return true;
        }
    }
    return false;
}


// Helper function: check if cylinder intersects with a bounding box
// Returns true only if the cylinder (not a capsule) intersects the box
bool Foam::cylinderCutToCell::cylinderIntersectsBoundBox
(
    const boundBox& bb,
    const vector& axis,
    const vector& axisDir,
    const scalar magAxis,
    const scalar rad2
) const
{
    // Check if the cylinder axis passes through (or close to) the bounding box
    // We need to ensure the intersection is within the cylinder length (not beyond ends)

    const point& boxMin = bb.min();
    const point& boxMax = bb.max();
    const point boxCenter = bb.midpoint();
    const scalar boxRadius = mag(boxMax - boxMin) * 0.5;

    // Find projection of box center onto cylinder axis
    vector d = boxCenter - p1_;
    scalar t = (d & axisDir);  // projection along normalized axis

    // Check if the box is completely outside the cylinder's axial range
    // Consider the box could extend into the cylinder range by boxRadius
    if (t < -boxRadius || t > magAxis + boxRadius)
    {
        return false;
    }

    // Clamp t to [0, magAxis] to find closest point on axis within cylinder
    scalar tClamped = max(scalar(0), min(t, magAxis));

    // Closest point on axis (within cylinder length)
    point closestOnAxis = p1_ + tClamped * axisDir;

    // Distance from closest point to box center
    scalar dist2 = magSqr(closestOnAxis - boxCenter);

    // If distance is less than radius + box diagonal/2, they might intersect
    if (dist2 <= sqr(radius_ + boxRadius))
    {
        return true;
    }

    return false;
}


void Foam::cylinderCutToCell::combine(topoSet& set, const bool add) const
{
    const vector axis = p2_ - p1_;
    const scalar rad2 = sqr(radius_);
    const scalar magAxis2 = magSqr(axis);
    const scalar magAxis = Foam::sqrt(magAxis2);
    const vector axisDir = axis / (magAxis + VSMALL);

    const pointField& pts = mesh_.points();
    const cellList& cells = mesh_.cells();
    const faceList& faces = mesh_.faces();

    forAll(cells, celli)
    {
        bool selected = false;

        // First, quick check: does any vertex of the cell lie inside the cylinder?
        const cell& c = cells[celli];
        labelHashSet cellPoints(c.size() * 4);

        // Collect all unique points of this cell
        forAll(c, facei)
        {
            const face& f = faces[c[facei]];
            forAll(f, pointi)
            {
                cellPoints.insert(f[pointi]);
            }
        }

        // Check if any vertex is inside the cylinder
        forAllConstIter(labelHashSet, cellPoints, iter)
        {
            if (pointInCylinder(pts[iter.key()], axis, magAxis2, rad2))
            {
                selected = true;
                break;
            }
        }

        // If no vertex inside, check if cylinder axis intersects cell bounding box
        if (!selected)
        {
            // Build bounding box for this cell - collect points first
            pointField cellPts(cellPoints.size());
            label pi = 0;
            forAllConstIter(labelHashSet, cellPoints, iter)
            {
                cellPts[pi++] = pts[iter.key()];
            }
            boundBox cellBB(cellPts, false);

            if (cylinderIntersectsBoundBox(cellBB, axis, axisDir, magAxis, rad2))
            {
                // More detailed check: find the closest point on cylinder axis
                // to the cell center and verify it's within cylinder bounds
                const point& cellCtr = mesh_.cellCentres()[celli];
                vector d = cellCtr - p1_;
                scalar t = (d & axisDir);
                
                // Only proceed if cell center projection is within cylinder length
                // (with some tolerance based on cell size)
                scalar cellRadius = mag(cellBB.max() - cellBB.min()) * 0.5;
                if (t >= -cellRadius && t <= magAxis + cellRadius)
                {
                    // Clamp to cylinder range
                    scalar tClamped = max(scalar(0), min(t, magAxis));
                    point closestOnAxis = p1_ + tClamped * axisDir;

                    // Check radial distance from axis to cell center
                    scalar radialDist2 = magSqr(cellCtr - closestOnAxis);
                    
                    // Select if within radius + cell size tolerance
                    if (radialDist2 <= sqr(radius_ + cellRadius))
                    {
                        selected = true;
                    }
                }
            }
        }

        if (selected)
        {
            addOrDelete(set, celli, add);
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::cylinderCutToCell::cylinderCutToCell
(
    const polyMesh& mesh,
    const vector& p1,
    const vector& p2,
    const scalar radius
)
:
    topoSetSource(mesh),
    p1_(p1),
    p2_(p2),
    radius_(radius)
{}


Foam::cylinderCutToCell::cylinderCutToCell
(
    const polyMesh& mesh,
    const dictionary& dict
)
:
    topoSetSource(mesh),
    p1_(dict.lookup("p1")),
    p2_(dict.lookup("p2")),
    radius_(readScalar(dict.lookup("radius")))
{}


Foam::cylinderCutToCell::cylinderCutToCell
(
    const polyMesh& mesh,
    Istream& is
)
:
    topoSetSource(mesh),
    p1_(checkIs(is)),
    p2_(checkIs(is)),
    radius_(readScalar(checkIs(is)))
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::cylinderCutToCell::~cylinderCutToCell()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::cylinderCutToCell::applyToSet
(
    const topoSetSource::setAction action,
    topoSet& set
) const
{
    if ((action == topoSetSource::NEW) || (action == topoSetSource::ADD))
    {
        Info<< "    Adding cells intersecting with cylinder, with p1 = "
            << p1_ << ", p2 = " << p2_ << " and radius = " << radius_ << endl;

        combine(set, true);
    }
    else if (action == topoSetSource::DELETE)
    {
        Info<< "    Removing cells intersecting with cylinder, with p1 = "
            << p1_ << ", p2 = " << p2_ << " and radius = " << radius_ << endl;

        combine(set, false);
    }
}


// ************************************************************************* //
