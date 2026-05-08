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

Application
    dfAutoPatch

Description
    An extension of autoPatch to work with polyhedral meshes.
    This utility divides external faces into patches based on (user supplied) feature
    angle and the source patches can be limited to user supplied patches if desired.
    The resulting patches are renamed to <originalName>_autoN if only a subset of 
    patches is processed, otherwise to autoN.

    Example usage:
    dfAutoPatch -onlyPatches "(box)" 45

Authorship
    Author: Teng Zhang @ AISI, <ztnuaa0211@gmail.com>
    Date: 2025-10-22
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "polyMesh.H"
#include "Time.H"
#include "boundaryMesh.H"
#include "repatchPolyTopoChanger.H"
#include "unitConversion.H"
#include "OFstream.H"
#include "ListOps.H"
#include "wordList.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Get all feature edges.
void collectFeatureEdges(const boundaryMesh& bMesh, labelList& markedEdges)
{
    markedEdges.setSize(bMesh.mesh().nEdges());

    label markedI = 0;

    forAll(bMesh.featureSegments(), i)
    {
        const labelList& segment = bMesh.featureSegments()[i];

        forAll(segment, j)
        {
            label featEdgeI = segment[j];

            label meshEdgeI = bMesh.featureToEdge()[featEdgeI];

            markedEdges[markedI++] = meshEdgeI;
        }
    }
    markedEdges.setSize(markedI);
}



int main(int argc, char *argv[])
{
    #include "addOverwriteOption.H"
    argList::noParallel();
    argList::validArgs.append("feature angle[0-180]");
    argList::addOption
    (
        "onlyPatches",
        "wordList of boundary patch names to be repatched"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    runTime.functionObjects().off();
    #include "createPolyMesh.H"
    const word oldInstance = mesh.pointsInstance();

    wordList onlyPatches;

    if (args.optionFound("onlyPatches"))
    {
        onlyPatches = wordList(args.optionLookup("onlyPatches")());

        Info<< "Limiting autoPatch to patches: " << onlyPatches
            << nl << endl;
    }

    Info<< "Mesh read in = "
        << runTime.cpuTimeIncrement()
        << " s\n" << endl << endl;


    const scalar featureAngle = args.argRead<scalar>(1);
    const bool overwrite      = args.optionFound("overwrite");

    const scalar minCos = Foam::cos(degToRad(featureAngle));

    Info<< "Feature:" << featureAngle << endl
        << "minCos :" << minCos << endl
        << endl;

    //
    // Use boundaryMesh to reuse all the featureEdge stuff in there.
    //

    boundaryMesh bMesh;
    bMesh.read(mesh);

    // Set feature angle (calculate feature edges)
    bMesh.setFeatureEdges(minCos);

    // Collect all feature edges as edge labels
    labelList markedEdges;

    collectFeatureEdges(bMesh, markedEdges);



    const labelList& boundaryFaces = bMesh.meshFace();
    const polyBoundaryMesh& originalBoundary = mesh.boundaryMesh();
    const label nBoundaryFaces = boundaryFaces.size();

    // (new) patch ID for every face in mesh.
    labelList patchIDs(nBoundaryFaces, -1);
    boolList processFace(nBoundaryFaces, false);
    labelList originalPatchIDs(nBoundaryFaces, -1);

    // Track per-patch suffix when only a subset is processed.
    labelList patchSuffixCount(originalBoundary.size(), 0);

    const bool limitToPatches = onlyPatches.size() != 0;

    // Determine which faces are allowed to be repatched.
    forAll(boundaryFaces, facei)
    {
        const label meshFacei = boundaryFaces[facei];
        const label patchi = originalBoundary.whichPatch(meshFacei);
        const word& patchName = originalBoundary[patchi].name();

        originalPatchIDs[facei] = patchi;
        patchIDs[facei] = patchi;

        if (!limitToPatches || findIndex(onlyPatches, patchName) != -1)
        {
            patchIDs[facei] = -1;
            processFace[facei] = true;
        }
    }

    if (limitToPatches)
    {
        label selectedFaceCount = 0;

        forAll(processFace, facei)
        {
            if (processFace[facei])
            {
                ++selectedFaceCount;
            }
        }

        if (!selectedFaceCount)
        {
            Info<< "No faces found for the supplied onlyPatches option." << nl
                << endl;
        }
    }

    //
    // Fill patchIDs with values for every face by floodfilling without
    // crossing feature edge.
    //

    // Current patch number.
    label newPatchi = bMesh.patches().size();

    label suffix = 0;

    while (true)
    {
        // Find first unset face respecting the optional patch filter.
        label unsetFacei = -1;

        forAll(patchIDs, facei)
        {
            if (patchIDs[facei] == -1 && processFace[facei])
            {
                unsetFacei = facei;
                break;
            }
        }

        if (unsetFacei == -1)
        {
            // All faces have patchID set. Exit.
            break;
        }

        // Found unset face. Create patch for it.
        word patchName;
        if (limitToPatches)
        {
            const label basePatchi = originalPatchIDs[unsetFacei];
            const word& baseName = originalBoundary[basePatchi].name();
            label& nextSuffix = patchSuffixCount[basePatchi];

            do
            {
                patchName = baseName + "_auto" + name(nextSuffix++);
            }
            while (bMesh.findPatchID(patchName) != -1);
        }
        else
        {
            do
            {
                patchName = "auto" + name(suffix++);
            }
            while (bMesh.findPatchID(patchName) != -1);
        }

        bMesh.addPatch(patchName);

        bMesh.changePatchType(patchName, "patch");


        // Fill visited with all faces reachable from unsetFacei.
        boolList visited(nBoundaryFaces, false);

        bMesh.markFaces(markedEdges, unsetFacei, visited);


        // Assign all visited faces to current patch
        label nVisited = 0;

        forAll(visited, facei)
        {
            if (visited[facei] && processFace[facei])
            {
                nVisited++;

                patchIDs[facei] = newPatchi;
            }
        }

        Info<< "Assigned " << nVisited << " faces to patch " << patchName
            << endl << endl;

        newPatchi++;
    }



    const PtrList<boundaryPatch>& patches = bMesh.patches();

    // Create new list of patches with old ones first
    List<polyPatch*> newPatchPtrList(patches.size());

    newPatchi = 0;

    // Copy old patches
    forAll(mesh.boundaryMesh(), patchi)
    {
        const polyPatch& patch = mesh.boundaryMesh()[patchi];

        newPatchPtrList[newPatchi] =
            patch.clone
            (
                mesh.boundaryMesh(),
                newPatchi,
                patch.size(),
                patch.start()
            ).ptr();

        newPatchi++;
    }

    // Add new ones with empty size.
    for (label patchi = newPatchi; patchi < patches.size(); patchi++)
    {
        const boundaryPatch& bp = patches[patchi];

        newPatchPtrList[newPatchi] = polyPatch::New
        (
            polyPatch::typeName,
            bp.name(),
            0,
            mesh.nFaces(),
            newPatchi,
            mesh.boundaryMesh()
        ).ptr();

        newPatchi++;
    }

    if (!overwrite)
    {
        runTime++;
    }


    // Change patches
    repatchPolyTopoChanger polyMeshRepatcher(mesh);
    polyMeshRepatcher.changePatches(newPatchPtrList);


    // Change face ordering

    // Since bMesh read from mesh there is one to one mapping so we don't
    // have to do the geometric stuff.
    forAll(patchIDs, facei)
    {
        label meshFacei = boundaryFaces[facei];

        polyMeshRepatcher.changePatchID(meshFacei, patchIDs[facei]);
    }

    polyMeshRepatcher.repatch();

    // Write resulting mesh
    if (overwrite)
    {
        mesh.setInstance(oldInstance);
    }

    // Set the precision of the points data to 10
    IOstream::defaultPrecision(max(10u, IOstream::defaultPrecision()));

    mesh.write();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
