#pragma once


#include "file_dataset.hh"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace bfs = boost::filesystem;

struct ParsedEntry {
    BoundingBoxList boxes;
    std::string imgPath;
};

class DTLDataset : public FileDataset {
public:
    enum class Mode {
        Train = 0,
        Val
    };

    DTLDataset(bfs::path basePath, Mode mode);
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) override;

private:
    cv::Mat parseGt(const BoundingBoxList& boxes, const cv::Size imageSize);
    bool debugImgWritten_{false};

    bool m_extractBoundingboxes;
    bfs::path m_labelFile;
    std::map<std::string, ParsedEntry> m_labels;

    /* TODO */
    const std::map<std::string, int32_t> m_instanceDict {
            {"traffic light front relevant", 9},
            {"traffic light front irrelevant", 10},
            {"traffic light left", 11},
            {"traffic light right", 12},
            {"traffic light back", 13},
    };
};