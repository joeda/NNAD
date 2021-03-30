#include "dtld_dataset.hh"

#include <sstream>

#include "utils.hh"

namespace {
    std::string convertToLocalPath(const bfs::path& basePath, const std::string& imgPath) {
        std::vector<std::string> origins = {"/scr/cmtcde8u013/fs2/DTLD_final/", "/scratch/fs2/DTLD_final/"};
        auto imgPathCopy = imgPath;
        for (const auto& ss : origins) {
            if (imgPath.rfind(ss, 0) == 0) {
                imgPathCopy.replace(0, ss.length(), basePath.string());
                return imgPathCopy;
            }
        }
        CHECK(false, "Invalid base path in DTLD");
    }

    std::string convertLabel(const std::string& label) {

    }

    std::map<std::string, BoundingBoxList> parseYaml(const bfs::path &yamlFile, const bfs::path& basePath, const int64 baseId) {
        YAML::Node labels = YAML::LoadFile(yamlFile.string());
        CHECK(labels.IsSequence(), "YAML root not a sequence");
        for (auto it = labels.begin(); it != labels.end(); ++it) {
            auto key = convertToLocalPath(basePath, (*it)["path"].as<std::string>());
            BoundingBoxList boxes;
            auto objectId = baseId;
            for (auto oit = (*it)["objects"].begin(); oit != (*it)["objects"].end(); ++oit) {
                auto x = oit->operator[]("x").as<int>();
                auto y = oit->operator[]("y").as<int>();
                auto w = oit->operator[]("width").as<int>();
                auto h = oit->operator[]("height").as<int>();
                BoundingBox boundingBox;
                boundingBox.id = objectId++;
                boundingBox.cls = m_instanceDict.at(cls);
                boundingBox.x1 = x;
                boundingBox.x2 = x + w;
                boundingBox.y1 = y;
                boundingBox.y2 = y + h;
                boxes.boxes.push_back(boundingBox);
            }
        }
    }
}

DTLDataset::DTLDataset(bfs::path basePath, Mode mode) {
    switch (mode) {
        case Mode::Train:
            m_labelFile = basePath / bfs::path("DTLD_Labels") / bfs::path("DTLD_train.yaml");
            m_extractBoundingboxes = true;
            break;
        case Mode::Val:
            m_labelFile = basePath / bfs::path("DTLD_Labels") / bfs::path("DTLD_test.yaml");
            m_extractBoundingboxes = true;
            break;
        default:
            CHECK(false, "Unknown mode!");
    }

    for (auto &entry : bfs::recursive_directory_iterator(m_groundTruthPath)) {
        if (entry.path().extension() == ".txt") {
            auto relativePath = bfs::relative(entry.path(), m_groundTruthPath);
            std::string key = relativePath.string();
            key = key.substr(0, key.length() - entry.path().extension().string().length());
            m_keys.push_back(key);
        }
    }
    std::sort(m_keys.begin(), m_keys.end());
}

std::shared_ptr<DatasetEntry> KittiDataset::get(std::size_t i) {
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();
    auto leftImgPath = m_leftImgPath / bfs::path(key + ".png");
    cv::Mat leftImg = cv::imread(leftImgPath.string());
    CHECK(leftImg.data, "Failed to read image " + leftImgPath.string());
    result->input.left = toFloatMat(leftImg);
    if (m_extractBoundingboxes) {
        auto gtPath = m_groundTruthPath / bfs::path(key + ".txt");
        std::ifstream gtFs(gtPath.string());
        auto[bbDontCareAreas, bbList] = parseGt(gtFs, result->input.left.size());
        result->gt.bbDontCareAreas = bbDontCareAreas;
        result->gt.bbList = bbList;
    }
    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = true;
    result->metadata.horizontalFov = 90.0;
    result->metadata.key = key;
    return result;
}

std::tuple<cv::Mat, BoundingBoxList> KittiDataset::parseGt(std::ifstream &gtFs, cv::Size imageSize) {
    cv::Mat bbDontCareImg(imageSize, CV_32SC1, cv::Scalar(m_boundingBoxValidLabel));
    BoundingBoxList bbList;
    bbList.valid = true;
    bbList.width = imageSize.width;
    bbList.height = imageSize.height;

    std::string line;
    std::stringstream splitter;
    int64_t objectId = getRandomId();
    while (std::getline(gtFs, line)) {
        splitter << line;
        std::string cls;
        splitter >> cls;
        double ddummy;
        int idummy;
        splitter >> ddummy;
        splitter >> idummy;
        splitter >> ddummy;
        double x1, y1, x2, y2;
        splitter >> x1 >> y1 >> x2 >> y2;
        splitter.clear();
        if (m_instanceDict.count(cls) == 0) {
            /* Draw don't care image for bounding boxes. */
            cv::rectangle(bbDontCareImg, cv::Rect(x1, y1, x2 - x1, y2 - y1),
                          cv::Scalar(m_boundingBoxDontCareLabel), -1);
        } else {
            /* Generate bounding box list */
            BoundingBox boundingBox;
            boundingBox.id = objectId++;
            boundingBox.cls = m_instanceDict.at(cls);
            boundingBox.x1 = x1;
            boundingBox.x2 = x2;
            boundingBox.y1 = y1;
            boundingBox.y2 = y2;
            bbList.boxes.push_back(boundingBox);
        }
    }

    return {bbDontCareImg, bbList};
}
