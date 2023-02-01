/*
*	Copyright (C) 2019 by
*       Shenglan Du (dushenglan940128@163.com)
*       Liangliang Nan (liangliang.nan@gmail.com)
*       3D Geoinformation, TU Delft, https://3d.bk.tudelft.nl
*
*	This file is part of AdTree, which implements the 3D tree
*   reconstruction method described in the following paper:
*   -------------------------------------------------------------------------------------
*       Shenglan Du, Roderik Lindenbergh, Hugo Ledoux, Jantien Stoter, and Liangliang Nan.
*       AdTree: Accurate, Detailed, and Automatic Modeling of Laser-Scanned Trees.
*       Remote Sensing. 2019, 11(18), 2074.
*   -------------------------------------------------------------------------------------
*   Please consider citing the above paper if you use the code/program (or part of it).
*
*	AdTree is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License Version 3
*	as published by the Free Software Foundation.
*
*	AdTree is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "skeleton.h"

#include <easy3d/core/types.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/core/surface_mesh.h>
#include <easy3d/core/graph.h>
#include <easy3d/util/dialogs.h>
#include <easy3d/util/file_system.h>
#include <easy3d/fileio/point_cloud_io.h>
#include <easy3d/fileio/graph_io.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/algo/remove_duplication.h>

#include <iostream>

#define MIN_PARAMETERS 3

using namespace boost;
using namespace easy3d;

// Export skeleton
bool save_skeleton(const std::string& skeleton_file, PointCloud *cloud, Skeleton *skeleton_) {

    // save skeleton model
    const ::Graph& skeleton = skeleton_->get_simplified_skeleton();
    // const ::Graph& skeleton = skeleton_->get_smoothed_skeleton();

    if (boost::num_edges(skeleton) == 0) {
        std::cerr << "skeleton has 0 edges" << std::endl;
        return false;
    }

    // convert the boost graph to Graph (avoid modifying easy3d's GraphIO, or writing IO for boost graph)

    std::unordered_map<SGraphVertexDescriptor, easy3d::Graph::Vertex>  vvmap;
    easy3d::Graph g;

    auto vts = boost::vertices(skeleton);
    for (SGraphVertexIterator iter = vts.first; iter != vts.second; ++iter) {
        SGraphVertexDescriptor vd = *iter;
        if (boost::degree(vd, skeleton) != 0 ) { // ignore isolated vertices
            const vec3& vp = skeleton[vd].cVert;
            vvmap[vd] = g.add_vertex(vp);
        }
    }

    auto egs = boost::edges(skeleton);
    for (SGraphEdgeIterator iter = egs.first; iter != egs.second; ++iter) {
        SGraphVertexDescriptor s = boost::source(*iter, skeleton);
        SGraphVertexDescriptor t = boost::target(*iter, skeleton);
        g.add_edge(vvmap[s], vvmap[t]);
    }

    auto offset = cloud->get_model_property<dvec3>("translation");
    if (offset) {
        auto prop = g.model_property<dvec3>("translation");
        prop[0] = offset[0];
    }

    if (GraphIO::save(skeleton_file, &g)) {
        return true;
    }
    else
        return false;

}

// returns the number of processed input files.
int batch_reconstruct(std::string& xyz_file, const std::string& output_file) {

    std::cout << "processing xyz_file: " << xyz_file << std::endl;

    // load point_cloud
    PointCloud *cloud = PointCloudIO::load(xyz_file);
    if (cloud) {
        std::cout << "cloud loaded. num points: " << cloud->n_vertices() << std::endl;

        // compute bbox
        Box3 box;
        auto points = cloud->get_vertex_property<vec3>("v:point");
        for (auto v : cloud->vertices())
            box.add_point(points[v]);

        // remove duplicated points
        const float threshold = box.diagonal() * 0.001f;
        const auto &points_to_remove = RemoveDuplication::apply(cloud, threshold);
        for (auto v : points_to_remove)
            cloud->delete_vertex(v);
        cloud->garbage_collection();
        std::cout << "removed too-close points. num points: " << cloud->n_vertices() << std::endl;
    }
    else {
        std::cerr << "failed to load point cloud from '" << xyz_file << "'" << std::endl;
        return EXIT_FAILURE;
    }

    // reconstruct branches
    SurfaceMesh *mesh = new SurfaceMesh;
    // const std::string &branch_filename = file_system::base_name(cloud->name()) + "_branches.obj";
    // mesh->set_name(branch_filename);

    // Construct skeleton
    Skeleton *skeleton = new Skeleton();
    bool status = skeleton->reconstruct_branches(cloud, mesh);
    if (!status) {
        std::cerr << "failed in reconstructing branches" << std::endl;
        return EXIT_FAILURE;
    }

    // save skeleton model
    const std::string skeleton_file = output_file;
    if (save_skeleton(skeleton_file, cloud, skeleton)) {
        std::cout << "model of skeleton saved to: " << skeleton_file << std::endl;
    } 
    else
        std::cerr << "failed in saving the model of branches" << std::endl;
        return EXIT_FAILURE;


    // // copy translation property from point_cloud to surface_mesh
    // SurfaceMesh::ModelProperty<dvec3> prop = mesh->add_model_property<dvec3>("translation");
    // prop[0] = cloud->get_model_property<dvec3>("translation")[0];

    // // save branches model
    // const std::string branch_file = output_folder + "/" + branch_filename;
    // if (SurfaceMeshIO::save(branch_file, mesh)) {
    //     std::cout << "model of branches saved to: " << branch_file << std::endl;
    // }
    // else
    //     std::cerr << "failed in saving the model of branches" << std::endl;
    
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {

    if (argc < MIN_PARAMETERS) 
    {
        std::cerr << "no valid input found." << std::endl;
        return EXIT_FAILURE;
    }

    std::string first_arg(argv[1]);
    std::string second_arg(argv[2]);
    if (file_system::is_file(second_arg)) 
    {
        std::cerr << "second argument cannot be an existing file." << std::endl;
        return EXIT_FAILURE;
    }
    else 
    {
        std::string output_file = second_arg;
        if (file_system::is_file(first_arg)) 
        {
            std::string cloud_file = first_arg;
            if (batch_reconstruct(cloud_file, output_file) > 0) {
                return EXIT_SUCCESS;
            }
        } 
        else 
        {
            std::cerr << "unknown first argument (expecting a point cloud file in *.xyz format)." << std::endl;
            return EXIT_FAILURE;
        }
    }
}
