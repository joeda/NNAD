#pragma once


#include "file_dataset.hh"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace bfs = boost::filesystem;

class DTLDataset : public FileDataset {
public:
    enum class Mode {
        Train = 0,
        Val
    };

    DTLDataset(bfs::path basePath, Mode mode);
    virtual std::shared_ptr<DatasetEntry> get(std::size_t i) override;

private:
    //std::tuple<cv::Mat, BoundingBoxList> parseGt(std::ifstream &gtFs, cv::Size imageSize);

    bool m_extractBoundingboxes;
    bfs::path m_labelFile;
    std::map<std::string, BoundingBoxList> m_labels;

    /* TODO */
    const std::map<std::string, int32_t> m_instanceDict {
            {"traffic light car relevant", 10},
            {"traffic light car irrelevant", 11},
            {"traffic light pedestrian", 12},
            {"traffic light bike", 13},
            {"traffic light other", 14},
    };
};