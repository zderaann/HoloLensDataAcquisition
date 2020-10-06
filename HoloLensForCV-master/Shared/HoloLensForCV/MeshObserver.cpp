#include "pch.h"
#include "MeshObserver.h"



    // from https://uwp.programmingpedia.net/en/tutorial/10131/how-to-get-current-datetime-in-cplusplus-uwp
    static Windows::Foundation::DateTime GetCurrentDateTime() {
        // Get the current system time
        SYSTEMTIME st;
        GetSystemTime(&st);

        // Convert it to something DateTime will understand
        FILETIME ft;
        SystemTimeToFileTime(&st, &ft);

        // Conversion to DateTime's long long is done vie ULARGE_INTEGER
        ULARGE_INTEGER ui;
        ui.LowPart = ft.dwLowDateTime;
        ui.HighPart = ft.dwHighDateTime;

        Windows::Foundation::DateTime currentDateTime;
        currentDateTime.UniversalTime = ui.QuadPart;
        return currentDateTime;
    }

    MeshObserver::MeshObserver() {
        m_count = 0;
        m_num = 0;
        m_maxTrianglesPerCubicMeter = 0;
    }


    MeshObserver::MeshObserver(int maxTrianglesPerCubicMeter,
        Windows::Perception::Spatial::SpatialLocator^ loc,
        Windows::Perception::Spatial::SpatialLocatorAttachedFrameOfReference^ refFr,
        Windows::Graphics::Holographic::HolographicSpace^ holographicSpace)
    {
        //m_surfaceObserver = ref new Windows::Perception::Spatial::Surfaces::SpatialSurfaceObserver;
        m_referenceFrame = refFr;
        m_holographicSpace = holographicSpace;
        m_spatialPerceptionAccessRequested = false;
        m_surfaceAccessAllowed = false;


        m_maxTrianglesPerCubicMeter = maxTrianglesPerCubicMeter;
        m_locator = loc;

        m_count = 0;
        m_num = 0;

    }

    void MeshObserver::SetArchiveFolder(Windows::Storage::StorageFolder^ archiveSourceFolder) {
        m_archiveSourceFolder = archiveSourceFolder;
    }

    void MeshObserver::GetSurfaces(Windows::Perception::Spatial::SpatialCoordinateSystem^ sys, Platform::String^ timestamp)
    {

        Windows::Graphics::Holographic::HolographicFrame^ holographicFrame = m_holographicSpace->CreateNextFrame();

        Windows::Graphics::Holographic::HolographicFramePrediction^ prediction = holographicFrame->CurrentPrediction;

        

        Windows::Perception::Spatial::SpatialCoordinateSystem^ currentCoordinateSystem = m_referenceFrame->GetStationaryCoordinateSystemAtTimestamp(prediction->Timestamp);

        // Only create a surface observer when you need to - do not create a new one each frame.
        if (m_surfaceObserver == nullptr)
        {
            // Initialize the Surface Observer using a valid coordinate system.
            if (!m_spatialPerceptionAccessRequested)
            {
                // The spatial mapping API reads information about the user's environment. The user must
                // grant permission to the app to use this capability of the Windows Holographic device.
                auto initSurfaceObserverTask = concurrency::create_task(Windows::Perception::Spatial::Surfaces::SpatialSurfaceObserver::RequestAccessAsync());
                initSurfaceObserverTask.then([this, currentCoordinateSystem](Windows::Perception::Spatial::SpatialPerceptionAccessStatus status)
                    {
                        switch (status)
                        {
                        case Windows::Perception::Spatial::SpatialPerceptionAccessStatus::Allowed:
                            m_surfaceAccessAllowed = true;
                            break;
                        default:
                            // Access was denied. This usually happens because your AppX manifest file does not declare the
                            // spatialPerception capability.
                            // For info on what else can cause this, see: http://msdn.microsoft.com/library/windows/apps/mt621422.aspx
                            dbg::trace(L"MeshObserver: Surface access denied\n");
                            m_surfaceAccessAllowed = false;
                            break;
                        }
                    });

                m_spatialPerceptionAccessRequested = true;
            }
        }

        if (m_surfaceAccessAllowed)
        {
            //Windows::Graphics::Holographic::HolographicCameraPose::TryGetVisibleFrustum;
            auto iter = prediction->CameraPoses->First();
            Windows::Graphics::Holographic::HolographicCameraPose^ cameraPose = iter->Current;
            Windows::Perception::Spatial::SpatialBoundingFrustum boundingFrustum = cameraPose->TryGetVisibleFrustum(currentCoordinateSystem)->Value;
            Windows::Perception::Spatial::SpatialBoundingVolume^ bounds = Windows::Perception::Spatial::SpatialBoundingVolume::FromFrustum(currentCoordinateSystem, boundingFrustum);

            // If status is Allowed, we can create the surface observer.
            if (m_surfaceObserver == nullptr)
            {
                // First, we'll set up the surface observer to use our preferred data formats.
                // In this example, a "preferred" format is chosen that is compatible with our precompiled shader pipeline.
                Windows::Perception::Spatial::Surfaces::SpatialSurfaceMeshOptions^ surfaceMeshOptions = ref new Windows::Perception::Spatial::Surfaces::SpatialSurfaceMeshOptions();
                Windows::Foundation::Collections::IVectorView<Windows::Graphics::DirectX::DirectXPixelFormat>^ supportedVertexPositionFormats = surfaceMeshOptions->SupportedVertexPositionFormats;
                unsigned int formatIndex = 0;
                if (supportedVertexPositionFormats->IndexOf(Windows::Graphics::DirectX::DirectXPixelFormat::R16G16B16A16IntNormalized, &formatIndex))
                {
                    surfaceMeshOptions->VertexPositionFormat = Windows::Graphics::DirectX::DirectXPixelFormat::R16G16B16A16IntNormalized;
                }
                Windows::Foundation::Collections::IVectorView<Windows::Graphics::DirectX::DirectXPixelFormat>^ supportedVertexNormalFormats = surfaceMeshOptions->SupportedVertexNormalFormats;
                if (supportedVertexNormalFormats->IndexOf(Windows::Graphics::DirectX::DirectXPixelFormat::R8G8B8A8IntNormalized, &formatIndex))
                {
                    surfaceMeshOptions->VertexNormalFormat = Windows::Graphics::DirectX::DirectXPixelFormat::R8G8B8A8IntNormalized;
                }

                m_surfaceObserver = ref new Windows::Perception::Spatial::Surfaces::SpatialSurfaceObserver();

            }

            if (m_surfaceObserver) {
                m_surfaceObserver->SetBoundingVolume(bounds);


                //GET SURFACES


                Windows::Foundation::Collections::IMapView<Platform::Guid, Windows::Perception::Spatial::Surfaces::SpatialSurfaceInfo^>^ const& surfaceCollection = m_surfaceObserver->GetObservedSurfaces();
                Windows::Perception::Spatial::Surfaces::SpatialSurfaceInfo^ info;

                // Process surface adds and updates.
                m_count = 0;
                for (const auto& pair : surfaceCollection)
                {
                    auto id = pair->Key;
                    auto surfaceInfo = pair->Value;
                    info = pair->Value;

                    GetSurfaceAsync(id, surfaceInfo, m_num, sys, timestamp);
                    
                }
                m_num++;

            }
        }
    }

    Platform::String^ MatrixToString(Windows::Foundation::Numerics::float4x4 matrix) {
        Platform::String^ m11 = Platform::String::Concat(matrix.m11, " ");
        Platform::String^ m12 = Platform::String::Concat(matrix.m12, " ");
        Platform::String^ m13 = Platform::String::Concat(matrix.m13, " ");
        Platform::String^ m14 = Platform::String::Concat(matrix.m14, ";");

        Platform::String^ m21 = Platform::String::Concat(matrix.m21, " ");
        Platform::String^ m22 = Platform::String::Concat(matrix.m22, " ");
        Platform::String^ m23 = Platform::String::Concat(matrix.m23, " ");
        Platform::String^ m24 = Platform::String::Concat(matrix.m24, ";");

        Platform::String^ m31 = Platform::String::Concat(matrix.m31, " ");
        Platform::String^ m32 = Platform::String::Concat(matrix.m32, " ");
        Platform::String^ m33 = Platform::String::Concat(matrix.m33, " ");
        Platform::String^ m34 = Platform::String::Concat(matrix.m34, ";");

        Platform::String^ m41 = Platform::String::Concat(matrix.m41, " ");
        Platform::String^ m42 = Platform::String::Concat(matrix.m42, " ");
        Platform::String^ m43 = Platform::String::Concat(matrix.m43, " ");
        Platform::String^ m44 = Platform::String::Concat(matrix.m44, "\n");

        Platform::String^ row1 = Platform::String::Concat(Platform::String::Concat(m11, m12), Platform::String::Concat(m13, m14));
        Platform::String^ row2 = Platform::String::Concat(Platform::String::Concat(m21, m22), Platform::String::Concat(m23, m24));
        Platform::String^ row3 = Platform::String::Concat(Platform::String::Concat(m31, m32), Platform::String::Concat(m33, m34));
        Platform::String^ row4 = Platform::String::Concat(Platform::String::Concat(m41, m42), Platform::String::Concat(m43, m44));

        return  Platform::String::Concat(Platform::String::Concat(row1, row2), Platform::String::Concat(row3, row4));
    }



    //SAVING MESH - DELA PROBLEMY
    void MeshObserver::saveMesh(int numOfVerts, int numOfFaces, Platform::String^ scale,
        Windows::Storage::Streams::IBuffer^ verts,
        Windows::Storage::Streams::IBuffer^ norms,
        Windows::Storage::Streams::IBuffer^ faces,
        size_t cnt,
        int num,
        Platform::IBox<Windows::Foundation::Numerics::float4x4>^ scsToWorld,
        Platform::String^ timestamp) {

        Platform::String^ preffix = Platform::String::Concat(num.ToString(), L"_");
        Platform::String^ name = Platform::String::Concat(preffix, timestamp);
        Platform::String^ suffix = Platform::String::Concat(L"_", cnt.ToString());
        Platform::String^ fullname = Platform::String::Concat(name, suffix);


        concurrency::create_task(m_archiveSourceFolder->CreateFileAsync(Platform::String::Concat(fullname, L"_info.txt"), Windows::Storage::CreationCollisionOption::GenerateUniqueName)).then([scale, numOfFaces, numOfVerts, fullname, scsToWorld](Windows::Storage::StorageFile^ file) {
            
            Platform::String^ str0 = Platform::String::Concat(scale, L"\n");
            Platform::String^ str1 = Platform::String::Concat(numOfVerts, L"\n");
            Platform::String^ str01 = Platform::String::Concat(str0, str1);
            Platform::String^ str2 = Platform::String::Concat(numOfFaces, L"\n");
            Platform::String^ str23;
            if (scsToWorld != nullptr)
                str23 = Platform::String::Concat(str2, MatrixToString(scsToWorld->Value));
            else
                str23 = Platform::String::Concat(str2, "0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0");

            Platform::String^ text = Platform::String::Concat(str01, str23);

            Windows::Storage::Streams::IBuffer^ buffer = Windows::Security::Cryptography::CryptographicBuffer::ConvertStringToBinary(text, Windows::Security::Cryptography::BinaryStringEncoding::Utf8);

            //WriteText dela problemy, WriteBuffer ne
            concurrency::create_task(Windows::Storage::FileIO::WriteBufferAsync(file, buffer)).then([fullname]() {});
            });
        
        concurrency::create_task(m_archiveSourceFolder->CreateFileAsync(Platform::String::Concat(fullname, L"_verts.txt"), Windows::Storage::CreationCollisionOption::GenerateUniqueName)).then([verts, fullname](Windows::Storage::StorageFile^ file) {
            concurrency::create_task(Windows::Storage::FileIO::WriteBufferAsync(file, verts)).then([fullname]() {});
            });

        concurrency::create_task(m_archiveSourceFolder->CreateFileAsync(Platform::String::Concat(fullname, L"_norms.txt"), Windows::Storage::CreationCollisionOption::GenerateUniqueName)).then([norms, fullname](Windows::Storage::StorageFile^ file) {
            concurrency::create_task(Windows::Storage::FileIO::WriteBufferAsync(file, norms)).then([fullname]() {});
            });

        concurrency::create_task(m_archiveSourceFolder->CreateFileAsync(Platform::String::Concat(fullname, L"_faces.txt"), Windows::Storage::CreationCollisionOption::GenerateUniqueName)).then([faces, fullname](Windows::Storage::StorageFile^ file) {
            concurrency::create_task(Windows::Storage::FileIO::WriteBufferAsync(file, faces)).then([fullname]() {});
            });

        dbg::trace(
            L"MeshObserver: Mesh %s written\n", fullname);
    }



    void MeshObserver::GetSurfaceAsync(Platform::Guid id, Windows::Perception::Spatial::Surfaces::SpatialSurfaceInfo^ newSurface, int num, Windows::Perception::Spatial::SpatialCoordinateSystem^ sys, Platform::String^ timestamp)
    {
        auto options = ref new Windows::Perception::Spatial::Surfaces::SpatialSurfaceMeshOptions();
        options->IncludeVertexNormals = true;

        size_t count = m_count;
        m_count++;

        // The level of detail setting is used to limit mesh complexity, by limiting the number
        // of triangles per cubic meter.
        auto createMeshTask = concurrency::create_task(newSurface->TryComputeLatestMeshAsync(m_maxTrianglesPerCubicMeter, options));
        auto processMeshTask = createMeshTask.then([this, id, num, sys, timestamp, count](Windows::Perception::Spatial::Surfaces::SpatialSurfaceMesh^ mesh)
            {
                if (mesh != nullptr)
                {

                    //------------------------------POSILANI MESHE---------------------------------//

                    int numOfVertices = mesh->VertexPositions->ElementCount;
                    int numOfFaces = mesh->TriangleIndices->ElementCount;
                    if (numOfVertices > 0) {




                        auto verBuff = mesh->VertexPositions->Data;
                        auto normBuff = mesh->VertexNormals->Data;
                        auto faceBuff = mesh->TriangleIndices->Data;

                        /*auto info = mesh->VertexPositions->Format;     //R16G16B6A16IntNormalized / R16G16B16A16Float
                        auto info1 = mesh->VertexNormals->Format;        //R8G8B8A8IntNormalized / R16G16B16A16Float
                        auto info2 = mesh->TriangleIndices->Format;      //R16Uint          */



                        Platform::String^ scalex = Platform::String::Concat(mesh->VertexPositionScale.x, ",");
                        Platform::String^ scaley = Platform::String::Concat(mesh->VertexPositionScale.y, ",");
                        Platform::String^ scalexy = Platform::String::Concat(scalex, scaley);
                        Platform::String^ scale = Platform::String::Concat(scalexy, mesh->VertexPositionScale.z);




                        Windows::Perception::Spatial::SpatialCoordinateSystem^ scs = mesh->CoordinateSystem;
                        Platform::IBox<Windows::Foundation::Numerics::float4x4>^ scsToWorld = scs->TryGetTransformTo(sys);


                        saveMesh(numOfVertices, numOfFaces, scale, verBuff, normBuff, faceBuff, count, num, scsToWorld, timestamp);                        

                        //m_count += 1;
                        //------------------------------POSILANI MESHE---------------------------------//
                    }
                }}, concurrency::task_continuation_context::use_current());
        

    }


    Windows::Perception::Spatial::SpatialLocator^ MeshObserver::GetSpatialLocator() {
        return m_locator;
    }

    void MeshObserver::SetSpatialLocator(Windows::Perception::Spatial::SpatialLocator^ loc) {
        m_locator = loc;
        m_referenceFrame = loc->CreateAttachedFrameOfReferenceAtCurrentHeading();
    }
