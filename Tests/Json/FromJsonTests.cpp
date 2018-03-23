// This file is part of the ACTS project.
//
// Copyright (C) 2017 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define BOOST_TEST_MODULE FromJson Tests
#include <boost/test/included/unit_test.hpp>
#include <climits>
#include <fstream>

#include "ACTS/Layers/DiscLayer.hpp"
#include "ACTS/Layers/ProtoLayer.hpp"
#include "ACTS/Plugins/JsonPlugin/lib/json.hpp"
#include "ACTS/Surfaces/RectangleBounds.hpp"
#include "ACTS/Surfaces/Surface.hpp"
#include "ACTS/Tools/LayerCreator.hpp"
#include "ACTS/Tools/SurfaceArrayCreator.hpp"
#include "ACTS/Utilities/Definitions.hpp"
#include "ACTS/Utilities/VariantData.hpp"

#include "ACTS/Plugins/JsonPlugin/FromJson.hpp"

namespace Acts {

namespace Test {

  std::vector<const Surface*>
  fullPhiTestSurfacesEC(size_t n     = 10,
                        double shift = 0,
                        double zbase = 0,
                        double r     = 10)
  {

    std::vector<const Surface*> res;

    double phiStep = 2 * M_PI / n;
    for (size_t i = 0; i < n; ++i) {

      double z = zbase + ((i % 2 == 0) ? 1 : -1) * 0.2;

      Transform3D trans;
      trans.setIdentity();
      trans.rotate(Eigen::AngleAxisd(i * phiStep + shift, Vector3D(0, 0, 1)));
      trans.translate(Vector3D(r, 0, z));

      auto bounds = std::make_shared<const RectangleBounds>(2, 1);

      auto           transptr = std::make_shared<const Transform3D>(trans);
      const Surface* srf      = new PlaneSurface(transptr, bounds);

      res.push_back(srf);  // use raw pointer
    }

    return res;
  }

  BOOST_AUTO_TEST_CASE(JsonLoader_load_test)
  {
    using json = nlohmann::json;
    using namespace std::string_literals;

    variant_map map;
    map["int"]    = 42;
    map["float"]  = 42.42;
    map["string"] = "hallo"s;
    variant_map object({{"key", "value"s}, {"other", "otherval"s}});
    map["object"] = object;
    variant_vector vector({true, "value"s});
    map["array"] = vector;

    std::string json_str = to_json(map, true);

    std::cout << json_str << std::endl;

    json json_parsed = json::parse(json_str);

    BOOST_TEST(map.get<int>("int") == json_parsed["int"].get<int>());
    BOOST_TEST(map.get<double>("float") == json_parsed["float"].get<double>());
    BOOST_TEST(map.get<std::string>("string")
               == json_parsed["string"].get<std::string>());

    BOOST_TEST(object.get<std::string>("key")
               == json_parsed["object"]["key"].get<std::string>());
    BOOST_TEST(object.get<std::string>("other")
               == json_parsed["object"]["other"].get<std::string>());

    BOOST_TEST(vector.get<bool>(0) == json_parsed["array"][0].get<bool>());
    BOOST_TEST(vector.get<std::string>(1)
               == json_parsed["array"][1].get<std::string>());

    variant_data var_from_json = from_json(json_parsed);
    std::cout << var_from_json << std::endl;

    variant_map map_from_json = boost::get<variant_map>(var_from_json);

    BOOST_TEST(map.get<int>("int") == map_from_json.get<int>("int"));
    BOOST_TEST(map.get<double>("float") == map_from_json.get<double>("float"));
    BOOST_TEST(map.get<std::string>("string")
               == map_from_json.get<std::string>("string"));

    variant_map object_json = map_from_json.get<variant_map>("object");
    BOOST_TEST(object.get<std::string>("key")
               == object_json.get<std::string>("key"));
    BOOST_TEST(object.get<std::string>("other")
               == object_json.get<std::string>("other"));

    variant_vector vector_json = map_from_json.get<variant_vector>("array");
    BOOST_TEST(vector.get<bool>(0) == vector_json.get<bool>(0));
    BOOST_TEST(vector.get<std::string>(1) == vector_json.get<std::string>(1));
  }

  BOOST_AUTO_TEST_CASE(JsonLoader_float_int_discrimination)
  {
    using json             = nlohmann::json;
    double       ref_value = 50.0;
    variant_data input     = ref_value;
    std::string  json_str  = to_json(input);
    std::cout << json_str << std::endl;
    json         json_parsed = json::parse(json_str);
    variant_data output      = from_json(json_parsed);
    double       value       = boost::get<double>(output);

    BOOST_TEST(ref_value == value);
  }

  BOOST_AUTO_TEST_CASE(JsonLoader_layer_load_test)
  {
    using json = nlohmann::json;
    using namespace std::string_literals;

    std::vector<const Surface*> surfaces;
    auto                        ringa = fullPhiTestSurfacesEC(30, 0, 0, 10);
    surfaces.insert(surfaces.end(), ringa.begin(), ringa.end());
    auto ringb = fullPhiTestSurfacesEC(30, 0, 0, 15);
    surfaces.insert(surfaces.end(), ringb.begin(), ringb.end());
    auto ringc = fullPhiTestSurfacesEC(30, 0, 0, 20);
    surfaces.insert(surfaces.end(), ringc.begin(), ringc.end());

    // ProtoLayer                 pl(surfaces);
    auto sac = std::make_shared<const SurfaceArrayCreator>(
        SurfaceArrayCreator::Config(),
        Acts::getDefaultLogger("SurfaceArrayCreator", Acts::Logging::VERBOSE));
    LayerCreator::Config cfg;
    cfg.surfaceArrayCreator = sac;
    LayerCreator lc(
        cfg, Acts::getDefaultLogger("LayerCreator", Acts::Logging::VERBOSE));

    std::shared_ptr<DiscLayer> layer = std::dynamic_pointer_cast<DiscLayer>(
        lc.discLayer(surfaces, equidistant, equidistant));

    const variant_data var_layer = layer->toVariantData();
    std::cout << (*layer->surfaceArray()) << std::endl;

    // std::cout << var_layer << std::endl;

    std::string json_string = to_json(var_layer, true);

    // check if nlohmann::json agrees we produced valid JSON
    auto json_parsed = json::parse(json_string);

    variant_data var_from_json = from_json(json_parsed);
    // std::cout << var_from_json << std::endl;

    auto layer2 = std::dynamic_pointer_cast<DiscLayer>(
        DiscLayer::create(var_from_json));
    std::cout << (*layer2->surfaceArray()) << std::endl;

    auto sa  = layer->surfaceArray();
    auto sa2 = layer2->surfaceArray();

    BOOST_TEST(sa);
    BOOST_TEST(sa2);

    BOOST_TEST(sa->transform().isApprox(sa2->transform()));

    for (const auto& srfRef : surfaces) {

      Vector3D ctr = srfRef->binningPosition(binR);

      std::vector<const Surface*> bc1 = sa->at(ctr);
      std::vector<const Surface*> bc2 = sa2->at(ctr);

      BOOST_TEST(bc1.size() == bc2.size());

      for (size_t i = 0; i < bc1.size(); i++) {
        auto srf1 = bc1.at(i);
        auto srf2 = bc2.at(i);

        // std::cout << srf1->transform().matrix() << std::endl <<
        // srf2->transform().matrix() << std::endl;
        BOOST_TEST(srf1->transform().isApprox(srf2->transform()));
      }
    }
  }

}  // end of namespace Test

}  // namespace Acts
