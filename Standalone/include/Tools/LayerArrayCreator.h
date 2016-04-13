///////////////////////////////////////////////////////////////////
// LayerArrayCreator.h, ACTS project
///////////////////////////////////////////////////////////////////

#ifndef ACTS_GEOMETRYTOOLS_LAYERARRAYCREATOR_H
#define ACTS_GEOMETRYTOOLS_LAYERARRAYCREATOR_H 1

#ifndef TRKDETDESCR_TAKESMALLERBIGGER
#define TRKDETDESCR_TAKESMALLERBIGGER
#define takeSmaller(current,test) current = current < test ? current : test
#define takeBigger(current,test)  current = current > test ? current : test
#define takeSmallerBigger(cSmallest, cBiggest, test) takeSmaller(cSmallest, test); takeBigger(cBiggest, test)
#endif

// Core module
#include "CoreInterfaces/AlgToolBase.h"
#include "Algebra/AlgebraDefinitions.h"
// Geometry module
#include "GeometryInterfaces/ILayerArrayCreator.h"
// STL
#include <algorithm>

namespace Acts {

    class Surface;
    class Layer;

    /** @class LayerArrayCreator

      The LayerArrayCreator is a simple Tool that helps to construct
      LayerArrays from std::vector of Acts::CylinderLayer, Acts::DiscLayer, Acts::PlaneLayer.

      It fills the gaps automatically with Acts::NavigationLayer to be processed easily in the
      Navigation of the Extrapolation process.

     @TODO Julia: make private tools private again after Gaudi update (bug in Gaudi), marked with //b
     
      @author Andreas.Salzburger@cern.ch   
     */

    class LayerArrayCreator : public AlgToolBase, virtual public ILayerArrayCreator {

      public:
        /** Constructor */
        LayerArrayCreator(const std::string&,const std::string&,const IInterface*);
        
        /** Destructor */
        virtual ~LayerArrayCreator();

        /** AlgTool and IAlgTool interface methods */
        static const InterfaceID& interfaceID() { return IID_ILayerArrayCreator; }

        /** AlgTool initialize method */
        virtual StatusCode initialize() override;
        
        /** AlgTool finalize method */
        virtual StatusCode finalize() override;

        /** LayerArraycreator interface method 
           - we assume the layer thickness to be used together with the binning value */
        LayerArray* layerArray(const LayerVector& layers, 
                               double min,
                               double max,
                               BinningType btype = arbitrary,
                               BinningValue bvalue = binX) const override; 
      
      private:
          Surface* createNavigationSurface(const Layer& layer, BinningValue bvalue, double offset) const;
    };

} // end of namespace

#endif // ACTS_GEOMETRYTOOLS_LAYERARRAYCREATOR_H

