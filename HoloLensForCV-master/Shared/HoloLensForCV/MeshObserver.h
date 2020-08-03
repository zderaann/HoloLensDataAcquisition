#pragma once

	class MeshObserver
	{
	public:
		MeshObserver();
		MeshObserver(int maxTrianglesPerCubicMeter,
					Windows::Perception::Spatial::SpatialLocator^ loc,
					Windows::Perception::Spatial::SpatialLocatorAttachedFrameOfReference^ refFr,
					Windows::Graphics::Holographic::HolographicSpace^ holographicSpace);

		    void GetSurfaceAsync(Platform::Guid id, 
								 Windows::Perception::Spatial::Surfaces::SpatialSurfaceInfo^ newSurface, 
								 int num, 
								 Windows::Perception::Spatial::SpatialCoordinateSystem^ sys,
								 Platform::String^ timestamp);

		void saveMesh(int numOfVerts, 
					int numOfFaces, 
					Platform::String^ scale,
					Windows::Storage::Streams::IBuffer^ verts,
					Windows::Storage::Streams::IBuffer^ norms,
					Windows::Storage::Streams::IBuffer^ faces,
					size_t cnt,
					int num,
					Platform::IBox<Windows::Foundation::Numerics::float4x4>^ scsToWorld,
					Platform::String^ timestamp);

		void GetSurfaces(Windows::Perception::Spatial::SpatialCoordinateSystem^ sys, Platform::String^ timestamp);

		void SetArchiveFolder(Windows::Storage::StorageFolder^ archiveSourceFolder);

		Windows::Perception::Spatial::SpatialLocator^ GetSpatialLocator();

		void SetSpatialLocator(Windows::Perception::Spatial::SpatialLocator^ loc);

	private:

		bool m_surfaceAccess;
		size_t m_count;
		int m_num;
		int m_maxTrianglesPerCubicMeter;

		bool m_spatialPerceptionAccessRequested;
		bool m_surfaceAccessAllowed;


		Windows::Perception::Spatial::Surfaces::SpatialSurfaceObserver^ m_surfaceObserver;
		Windows::Perception::Spatial::SpatialLocatorAttachedFrameOfReference^ m_referenceFrame;
		Windows::Perception::Spatial::SpatialLocator^ m_locator;
		Windows::Graphics::Holographic::HolographicSpace^ m_holographicSpace;

		Windows::Storage::StorageFolder^ m_archiveSourceFolder;
	};

