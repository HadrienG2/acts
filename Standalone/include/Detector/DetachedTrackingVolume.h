///////////////////////////////////////////////////////////////////
// DetachedTrackingVolume.h, ACTS project
///////////////////////////////////////////////////////////////////

#ifndef ACTS_DETECTOR_DETACHEDTRACKINGVOLUME_H
#define ACTS_DETECTOR_DETACHEDTRACKINGVOLUME_H 1

class MsgStream;

// Geometry module
#include "Detector/Layer.h"
#include "Detector/PlaneLayer.h"
#include "Surfaces/Surface.h"
#include "GeometryUtils/GeometrySignature.h"
// Core module
#include "Algebra/AlgebraDefinitions.h"

namespace Acts {
    
  class TrackingVolume;
  class Surface;
  
  typedef std::vector< LayerPtr > LayerVector;
  
  // master typedefs
  class DetachedTrackingVolume;
  typedef std::shared_ptr<const DetachedTrackingVolume> DetachedTrackingVolumePtr;
  typedef std::shared_ptr< const TrackingVolume>         TrackingVolumePtr;
  
  /**
   @class DetachedTrackingVolume
  
   Base Class for a navigation object (active/passive) in the Tracking geometry.

   @author Sarka.Todorova@cern.ch, Andreas.Salzburger@cern.ch
   
   */

  class DetachedTrackingVolume {
   
      /** Declare the IDetachedTrackingVolumeBuilder as a friend, to be able to change the volumelink */
      friend class TrackingVolume;
      friend class DetachedTrackingVolumeBuilder;
      friend class IDetachedTrackingVolumeBuilder;
            
      public:
        /** Factory Constructor */
        static DetachedTrackingVolumePtr create(const std::string& name,
                                                TrackingVolumePtr vol,
                                                LayerPtr layer = nullptr,
                                                LayerVector* multiLayer = nullptr) 
        { return DetachedTrackingVolumePtr(new DetachedTrackingVolume(name,vol,layer,multiLayer)); } 
                              
        /** Destructor*/
        ~DetachedTrackingVolume();
        
        /** returns the TrackingVolume */
        const TrackingVolume* trackingVolume() const;

        /** returns the Name */
        const std::string name() const;
                     
        /** moving object around */
        void move( Transform3D& shift) const;

        /** clone with shift */
        DetachedTrackingVolumePtr clone( std::string name, Transform3D& shift) const;

        /** returns layer representation */
        const Layer* layerRepresentation() const;

        /** returns (multi)layer representation */
        const LayerVector* multilayerRepresentation() const;

        /** sign the volume - the geometry builder has to do that */
        void sign(GeometrySignature signat, GeometryType geotype) const;
        
        /** return the Signature */
        GeometrySignature geometrySignature() const;

        /** return the Type */
        GeometryType geometryType() const;

        /** set the simplified calculable components */
        void saveConstituents(std::vector<std::pair<const Acts::Volume*,float> >* ) const;

        /** get the simplified calculable components */
        std::vector<std::pair<const Acts::Volume*,float> >* constituents() const;

	    /** alignment methods: set base transform / default argument to current transform */
	    void setBaseTransform( Transform3D* transf=0 ) const;
	
	    /** alignment methods: realign  / default argument to base transform */
	    void realign( Transform3D* transf=0 ) const;

    protected:
        /**Default Constructor*/
        DetachedTrackingVolume();
        
        /**Constructor with name & layer representation*/
        DetachedTrackingVolume(const std::string& name,
                               TrackingVolumePtr vol,
                               LayerPtr layer,
                               std::vector< LayerPtr >* multilayer);
        

    private:
                
        const std::string                        m_name;         
        TrackingVolumePtr                        m_trkVolume;
        LayerPtr                                 m_layerRepresentation;
        std::vector< LayerPtr >*                 m_multilayerRepresentation;
	    mutable Transform3D*                                         m_baseTransform;         // optional use (for alignment purpose) 
        mutable std::vector<std::pair<const Acts::Volume*,float> >*   m_constituents;  
        
                
  };

inline const TrackingVolume* DetachedTrackingVolume::trackingVolume() const { return m_trkVolume.get(); } 

inline const std::string DetachedTrackingVolume::name() const { return (m_name); }

inline const Layer* DetachedTrackingVolume::layerRepresentation() const { return m_layerRepresentation.get(); }

inline const LayerVector* DetachedTrackingVolume::multilayerRepresentation() const { return m_multilayerRepresentation; }
 
inline void DetachedTrackingVolume::saveConstituents(std::vector<std::pair<const Acts::Volume*,float> >* constituents ) const { m_constituents = constituents; } 

inline std::vector<std::pair<const Acts::Volume*,float> >* DetachedTrackingVolume::constituents() const
   { return m_constituents;}

} // end of namespace

#endif // ACTS_DETECTOR_DETACHEDTRACKINGVOLUME_H



