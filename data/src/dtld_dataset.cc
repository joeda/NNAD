#include "dtld_dataset.hh"

#include <sstream>

#include "utils.hh"

namespace {
    std::string convertToLocalPath(const bfs::path& basePath, const std::string& imgPath) {
        std::vector<std::string> origins = {"/scr/cmtcde8u013/fs2/DTLD_final", "/scratch/fs2/DTLD_final"};
        auto imgPathCopy = imgPath;
        for (const auto& ss : origins) {
            if (imgPath.rfind(ss, 0) == 0) {
                imgPathCopy.replace(0, ss.length(), basePath.string());
                return imgPathCopy;
            }
        }
        CHECK(false, "Invalid base path in DTLD");
        return "";
    }

    std::string convertLabel(const std::string& label) {
        if (label.size() != 6) {
            throw std::runtime_error("traffic light Label of size != 6");
        }
        if (label.at(0) == char('1')) { //facing front
            if (label.at(1) == char('1') || label.at(1)== char('3')) {
                return "traffic light front relevant";
            } else {
                return "traffic light front irrelevant";
            }
        } else if (label.at(0) == char('3')) { // facing left
            return "traffic light left";
        }else if (label.at(0) == char('4')) {
            return "traffic light right";
        }else if (label.at(0) == char('2')) {
            return "traffic light back";
        } else {
            throw std::runtime_error("Invalid traffic light direction");
        }
        return "";
    }

    std::map<std::string, ParsedEntry> parseYaml(const bfs::path &yamlFile, const bfs::path& basePath, const int64 baseId, const std::map<std::string, int32_t>& instMap) {
        std::cout << "Parsing " << yamlFile.string() << std::endl;
        YAML::Node labels = YAML::LoadFile(yamlFile.string());
        CHECK(labels.IsSequence(), "YAML root not a sequence");
        std::map<std::string, ParsedEntry> res;
        for (auto it = labels.begin(); it != labels.end(); ++it) {
            auto key = convertToLocalPath(basePath, (*it)["path"].as<std::string>());
            BoundingBoxList boxes;
            auto objectId = baseId;
            boxes.valid = true;
            boxes.width = 2048;
            boxes.height = 1024;
            for (auto oit = (*it)["objects"].begin(); oit != (*it)["objects"].end(); ++oit) {
                auto x = oit->operator[]("x").as<int>();
                auto y = oit->operator[]("y").as<int>();
                auto w = oit->operator[]("width").as<int>();
                auto h = oit->operator[]("height").as<int>();
                BoundingBox boundingBox;
                boundingBox.id = oit->operator[]("unique_id").as<int>();
                boundingBox.cls = instMap.at(convertLabel(oit->operator[]("class_id").as<std::string>()));
                boundingBox.x1 = x;
                boundingBox.x2 = x + w;
                boundingBox.y1 = y;
                boundingBox.y2 = y + h;
                boxes.boxes.push_back(boundingBox);
            }
            res.emplace(key, ParsedEntry{boxes, key});
        }
        std::cout << "finished YAML parse" << std::endl;
        return res;
    }

    cv::Mat readTiff(const std::string& path) {
        cv::Mat orig = cv::imread(path, cv::IMREAD_UNCHANGED);
        cv::Mat debayer;
        cv::cvtColor(orig, debayer, cv::COLOR_BayerGB2BGR);
        int channels = debayer.channels();
        int nRows = debayer.rows;
        int nCols = debayer.cols * channels;

        if (debayer.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }

        int i,j;
        ushort * p;
        for( i = 0; i < nRows; ++i) {
            p = debayer.ptr<ushort>(i);
            for ( j = 0; j < nCols; ++j) {
                p[j] = p[j] >> 4;
            }
        }
        debayer.convertTo(debayer, CV_8U);
        return debayer;
    }
}

DTLDataset::DTLDataset(bfs::path basePath, Mode mode) {
    switch (mode) {
        case Mode::Train:
            m_labelFile = basePath / bfs::path("DTLD_Labels") / bfs::path("DTLD_train.yml");
            m_extractBoundingboxes = true;
            break;
        case Mode::Val:
            m_labelFile = basePath / bfs::path("DTLD_Labels") / bfs::path("DTLD_test.yml");
            m_extractBoundingboxes = true;
            break;
        default:
            CHECK(false, "Unknown mode!");
    }
    std::cout << "Parsing YAML" << std::endl;
    m_labels = parseYaml(m_labelFile, basePath, getRandomId(), m_instanceDict);
    std::cout << "making keys" << std::endl;
    std::transform(m_labels.begin(), m_labels.end(), std::back_inserter(m_keys), [](const auto& pair) {return pair.first;});
    std::sort(m_keys.begin(), m_keys.end());
    std::cout << "finished reading" << std::endl;
}

std::shared_ptr<DatasetEntry> DTLDataset::get(std::size_t i) {
    CHECK(i < m_keys.size(), "Index out of range");
    auto key = m_keys[i];
    auto result = std::make_shared<DatasetEntry>();
    const auto& label = m_labels.at(key);
    auto leftImgPath = label.imgPath;
    auto leftImg = readTiff(leftImgPath);
    CHECK(leftImg.data, "Failed to read image " + leftImgPath);

    //REMOVE ME
    if (!debugImgWritten_) {
        cv::imwrite("/tmp/janosch_debug_img.png", leftImg);
        debugImgWritten_ = true;
    }
    result->input.left = toFloatMat(leftImg);
    result->input.prevLeft = toFloatMat(leftImg);
    if (m_extractBoundingboxes) {
        auto bbDontCareAreas = parseGt(label.boxes, result->input.left.size());
        result->gt.bbDontCareAreas = bbDontCareAreas;
        result->gt.bbList = label.boxes;
        result->gt.pixelwiseLabels = cv::Mat(result->input.left.size(), CV_32SC1, cv::Scalar(m_semanticDontCareLabel));
    }
    result->metadata.originalWidth = result->input.left.cols;
    result->metadata.originalHeight = result->input.left.rows;
    result->metadata.canFlip = true;
    result->metadata.horizontalFov = 50.0;
    result->metadata.key = key;
    return result;
}

cv::Mat DTLDataset::parseGt(const BoundingBoxList& boxes, const cv::Size imageSize) {
    cv::Mat bbDontCareImg(imageSize, CV_32SC1, cv::Scalar(m_boundingBoxValidLabel));
    for (const auto& bb : boxes.boxes) {
        if (std::find_if(m_instanceDict.begin(), m_instanceDict.end(), [&bb](const auto& mo) {return mo.second == bb.cls; }) != m_instanceDict.end()) {
            /* Draw don't care image for bounding boxes. */
            cv::rectangle(bbDontCareImg, cv::Rect(bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1),
                          cv::Scalar(m_boundingBoxDontCareLabel), -1);
        } else {

        }
    }

    return bbDontCareImg;
}
