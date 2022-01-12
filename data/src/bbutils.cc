/**************************************************************************
 * NNAD (Neural Networks for Automated Driving) training scripts          *
 * Copyright (C) 2019 FZI Research Center for Information Technology      *
 *                                                                        *
 * This program is free software: you can redistribute it and/or modify   *
 * it under the terms of the GNU General Public License as published by   *
 * the Free Software Foundation, either version 3 of the License, or      *
 * (at your option) any later version.                                    *
 *                                                                        *
 * This program is distributed in the hope that it will be useful,        *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 * GNU General Public License for more details.                           *
 *                                                                        *
 * You should have received a copy of the GNU General Public License      *
 * along with this program.  If not, see <https://www.gnu.org/licenses/>. *
 **************************************************************************/

#include <iostream>
#include <algorithm>
#include <cstring>

#include "bbutils.hh"

BBUtils::BBUtils(int width, int height, int scale) : m_width(width), m_height(height)
{
    CHECK(height % scale == 0, "Height must be divisible by scale");
    CHECK(width % scale == 0, "Width must be divisible by scale");
    std::map<int, std::vector<double>> depthLookup{{48, {4.180983730685947, 4.303287024768623, 4.169273560680486, 3.878170278534493, 3.8897873644064926, }},
                                                   {64, {4.18935816550426, 4.169692903024991, 3.9751562512867147, 3.8003547517060547, 3.93407416938058, }},
                                                   {96, {3.943458180728441, 4.0417153900934375, 3.866773667834607, 3.712700929183552, 3.622397628378822, }},
                                                   {128, {3.8301307623771845, 3.870271884361034, 3.802965600858119, 3.5074036433816067, 3.5000080575578316, }},
                                                   {192, {3.6590475761680157, 3.729644411637628, 3.6490860735688937, 3.48467407870976, 3.4783142800033717, }},
                                                   {256, {3.548948475342533, 3.5402489904628016, 3.536303926494152, 3.4202222067762253, 3.3185047388050775, }},
                                                   {384, {3.3305275535547287, 3.412503714697786, 3.4989482567582444, 3.4308081670291983, 3.2132220764571224, }},
                                                   {512, {3.2498298097859584, 3.2895190918530703, 3.394179972295157, 3.449556850211472, 3.156322806357458, }},
                                                   {768, {3.1526302333005223, 3.1632754249997226, 3.3056119980188363, 3.360806222767012, 3.1198801273294703, }},
                                                   {1024, {2.9656664752769353, 3.044875915327097, 3.240031774733845, 3.384034687867842, 3.1023801307608085, }},
                                                   {1536, {2.8385511236419414, 2.9153779721138178, 3.1264550558110087, 3.319111544785094, 2.94832549032906, }},
                                                   {2048, {2.720513781462783, 2.789608372313202, 3.0438124116819454, 3.27342951831106, 2.929197283860388, }},
                                                   {3072, {2.5674736798532605, 2.6461581229322038, 2.9533376696188256, 3.1629333180376817, 2.7714604821344504, }},
                                                   {4096, {2.02958582188163, 2.0852136830917525, 2.344136879054102, 2.561365332293856, 2.741279919937926, }}};

    int xSteps = width / scale;
    int ySteps = height / scale;
    int boxesPerPos = m_areas.size() * m_ratios.size();

    m_anchorBoxes = std::vector<AnchorBox>(xSteps * ySteps * boxesPerPos);

    int idx = 0;
    for (int j = 0; j < ySteps; ++j) {
        int yc = j * scale + 0.5 * scale;
        for (int i = 0; i < xSteps; ++i) {
            int xc = i * scale + 0.5 * scale;
            for (auto &area : m_areas) {
                for (size_t ratioIdx{0}; ratioIdx < m_ratios.size(); ++ratioIdx) {
                    const auto ratio = m_ratios.at(ratioIdx);
                    int boxWidth = std::pow(area * ratio, 0.5) * scale;
                    int boxHeight = std::pow(area / ratio, 0.5) * scale;
                    int x1 = xc - 0.5 * boxWidth;
                    int x2 = xc + 0.5 * boxWidth;
                    int y1 = yc - 0.5 * boxHeight;
                    int y2 = yc + 0.5 * boxHeight;

                    m_anchorBoxes[idx].xc = xc;
                    m_anchorBoxes[idx].yc = yc;
                    m_anchorBoxes[idx].x1 = x1;
                    m_anchorBoxes[idx].y1 = y1;
                    m_anchorBoxes[idx].x2 = x2;
                    m_anchorBoxes[idx].y2 = y2;
                    m_anchorBoxes[idx].depth = depthLookup.at(area * scale).at(ratioIdx);
                    ++idx;
                }
            }
        }
    }
}

void BBUtils::targetsFromBBList(std::shared_ptr<DatasetEntry> ds) const
{
    auto &list = ds->gt.bbList;
    auto &bbDontCareAreas = ds->gt.bbDontCareAreas;
    if (!list.valid) {
        setTargetsAsIgnore(ds);
        return;
    }
    CHECK(list.width == m_width, "The width of the box list does not match the expected value");
    CHECK(list.height == m_height, "The width of the box list does not match the expected value");

    auto targetBoxes = targetsFromSingleBBList(list.boxes);
    auto prevTargetBoxes = targetsFromSingleBBList(list.previousBoxes);

    /* We only mask the current target boxes here, because only these are used to train the detector. */
    if (!bbDontCareAreas.empty()) {
        for (std::size_t idx = 0; idx < m_anchorBoxes.size(); ++idx) {
            auto &anchorBox = m_anchorBoxes[idx];
            /* Mark boxes in unlabeled image regions as don't care */
            if (bbDontCareAreas.at<int32_t>(anchorBox.yc, anchorBox.xc) == 1) {
                targetBoxes[idx].objectness = Objectness::DONT_CARE;
                targetBoxes[idx].id = 0;
            }
        }
    }

    ds->gt.bbList.targets.push_back(targetBoxes);
    ds->gt.bbList.previousTargets.push_back(prevTargetBoxes);
}

std::vector<TargetBox> BBUtils::targetsFromSingleBBList(std::vector<BoundingBox> &boxes) const
{
    std::vector<double> currentIouAtTarget(m_anchorBoxes.size());
    std::vector<TargetBox> targetBoxes(m_anchorBoxes.size());
    std::fill(currentIouAtTarget.begin(), currentIouAtTarget.end(), 0.0);
    int boxId = 0;
    for (auto &box : boxes) {
        auto bbIou = anchorIou(box);
        bool boxWasAssigned = false;
        double maxIou = -1.0;
        double maxIouIdx = -1;
        ++boxId;
        for (std::size_t idx = 0; idx < m_anchorBoxes.size(); ++idx) {
            if (bbIou[idx] > maxIou) {
                maxIou = bbIou[idx];
                maxIouIdx = idx;
            }

            if (std::abs(bbIou[idx] - currentIouAtTarget[idx]) < 0.2) {
                /* The target is in the middle of two objects with high overlap.
                 * Disable it because otherwise the regression will not converge to one
                 * of the objects and the box might end up in the middle. */
                if (bbIou[idx] > 0.4) {
                    targetBoxes[idx].objectness = Objectness::NO_OBJECT;
                    targetBoxes[idx].id = 0;
                }
            } else if (bbIou[idx] > currentIouAtTarget[idx]) {
                if (bbIou[idx] > 0.5) {
                    boxToTarget(box, m_anchorBoxes[idx], targetBoxes[idx]);
                    targetBoxes[idx].id = boxId;
                    boxWasAssigned = true;
                } else if (bbIou[idx] > 0.4) {
                    targetBoxes[idx].objectness = Objectness::DONT_CARE;
                    targetBoxes[idx].id = 0;
                }
            }

            if (bbIou[idx] > currentIouAtTarget[idx]) {
                currentIouAtTarget[idx] = bbIou[idx];
            }
        }
        if (!boxWasAssigned) {
            if (maxIou > 0.4) {
                boxToTarget(box, m_anchorBoxes[maxIouIdx], targetBoxes[maxIouIdx]);
                targetBoxes[maxIouIdx].id = boxId;
                currentIouAtTarget[maxIouIdx] = bbIou[maxIouIdx];
                /* TODO: How to handle boxes that were overwritten? */
            } else {
                /* TODO: How to handle this? */
                //std::cout << "WARNING: We could not find a target for a box with size " << box.x2 - box.x1
                //          << " x " << box.y2 - box.y1 << " (max IoU " << bbIou[maxIouIdx] << ")" << std::endl;
            }
        }
    }
    return targetBoxes;
}

std::vector<BoundingBoxDetection> BBUtils::bbListFromTargets(VectorView<float>(objectnessScores),
                                                             VectorView<int64_t> objectClass,
                                                             VectorView<float> depth,
                                                             VectorView<float> regression,
                                                             VectorView<float> deltaRegression,
                                                             VectorView<float> embedding,
                                                             int embeddingLength, double threshold) const
{
    std::vector<BoundingBoxDetection> detectionList;
    for (std::size_t idx = 0; idx < m_anchorBoxes.size(); ++idx) {
        if (objectnessScores[idx] >= threshold) {
            TargetBoxDetection target;
            target.dxc = regression[4 * idx + 0];
            target.dyc = regression[4 * idx + 1];
            target.dw = regression[4 * idx + 2];
            target.dh = regression[4 * idx + 3];
            if (deltaRegression.size() > 0) {
                target.deltaPrevXc = deltaRegression[4 * idx + 0];
                target.deltaPrevYc = deltaRegression[4 * idx + 1];
                target.deltaPrevW = deltaRegression[4 * idx + 2];
                target.deltaPrevH = deltaRegression[4 * idx + 3];
            }
            target.cls = objectClass[idx];
            target.depth = depth[idx];
            target.objectnessScore = objectnessScores[idx];
            target.embedding.resize(embeddingLength);
            std::memcpy(&target.embedding[0], embedding.data() + embeddingLength * idx,
                        embeddingLength * sizeof(float));

            /* decode detection */
            BoundingBoxDetection detection;
            detection.score = objectnessScores[idx];
            detectionToBox(target, m_anchorBoxes[idx], detection);
            detectionList.push_back(detection);
        }
    }

    return detectionList;
}

std::size_t BBUtils::numAnchors() const
{
    return m_anchorBoxes.size();
}

template <typename B1, typename B2>
static inline double calculateIou(const B1 &b1, const B2 &b2)
{
    int maxX1 = std::max(b1.x1, b2.x1);
    int maxY1 = std::max(b1.y1, b2.y1);
    int minX2 = std::min(b1.x2, b2.x2);
    int minY2 = std::min(b1.y2, b2.y2);

    int b1Area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    int b2Area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);

    int interWidth = std::max(minX2 - maxX1, 0);
    int interHeight = std::max(minY2 - maxY1, 0);
    int interArea = interWidth * interHeight;
    int unionArea = b1Area + b2Area - interArea;
    if (unionArea == 0) {
        return 0.0;
    }
    return static_cast<double>(interArea) / static_cast<double>(unionArea);
}

void BBUtils::performNMS(std::vector<BoundingBoxDetection> &detectionList, double lowOverlapThreshold,
                         double highOverlapThreshold)
{
    struct DetectionIndex {
        enum class State {
            UNDECIDED = 0,
            PICKED,
            NOT_PICKED
        };

        int64_t idx;
        State state;
    };

    std::vector<DetectionIndex> idxs(detectionList.size());
    for (std::size_t i = 0; i < idxs.size(); ++i) {
        idxs[i].idx = i;
        idxs[i].state = DetectionIndex::State::UNDECIDED;
    }

    std::sort(idxs.begin(), idxs.end(), [&] (DetectionIndex &a, DetectionIndex &b) {
            return detectionList[a.idx].score < detectionList[b.idx].score;
        });

    for (int outerIdx = idxs.size() - 1; outerIdx >= 0; --outerIdx) {
        if (idxs[outerIdx].state != DetectionIndex::State::UNDECIDED) {
            continue;
        }
        idxs[outerIdx].state = DetectionIndex::State::PICKED;
        for (int innerIdx = outerIdx - 1; innerIdx >= 0; --innerIdx) {
            if (idxs[innerIdx].state == DetectionIndex::State::UNDECIDED) {
                bool isSame;
                double iou = calculateIou(detectionList[idxs[innerIdx].idx].box, detectionList[idxs[outerIdx].idx].box);
                if (iou > highOverlapThreshold) {
                    isSame = true;
                } else if (iou < lowOverlapThreshold) {
                    isSame = false;
                } else {
                    // decide based on feature distance
                    double featureDist = 0;
                    for (std::size_t j = 0; j < detectionList[idxs[innerIdx].idx].embedding.size(); ++j) {
                        featureDist += std::pow(detectionList[idxs[innerIdx].idx].embedding[j]
                                                - detectionList[idxs[outerIdx].idx].embedding[j], 2.0);
                    }
                    isSame = featureDist <= 1.2;
                }

                if (isSame) {
                    idxs[innerIdx].state = DetectionIndex::State::NOT_PICKED;
                }
            }
        }
    }

    std::sort(idxs.begin(), idxs.end(), [] (DetectionIndex &a, DetectionIndex &b) {
            return a.idx < b.idx;
        });

    int64_t i = 0;
    detectionList.erase(std::remove_if(detectionList.begin(), detectionList.end(), [&] (BoundingBoxDetection &) {
            return idxs[i++].state != DetectionIndex::State::PICKED;
        }), detectionList.end());
}

void BBUtils::setTargetsAsIgnore(std::shared_ptr<DatasetEntry> ds) const
{
    std::vector<TargetBox> targetBoxes(m_anchorBoxes.size());
    for (auto &target : targetBoxes) {
        target.objectness = Objectness::DONT_CARE;
    }
    ds->gt.bbList.targets.push_back(targetBoxes);
    ds->gt.bbList.previousTargets.push_back(targetBoxes);
}

std::vector<double> BBUtils::anchorIou(const BoundingBox &box) const
{
    std::vector<double> ious(m_anchorBoxes.size());
    for (std::size_t idx = 0; idx < m_anchorBoxes.size(); ++idx) {
        auto &anchorBox = m_anchorBoxes[idx];
        ious[idx] = calculateIou(box, anchorBox);
    }
    return ious;
}

void BBUtils::boxToTarget(const BoundingBox &box, const AnchorBox &anchor, TargetBox &target) const
{
    target.objectness = Objectness::OBJECT;
    target.cls = box.cls;
    target.depth = box.depth - anchor.depth;

    double anchorWidth = anchor.x2 - anchor.x1;
    double anchorHeight = anchor.y2 - anchor.y1;
    double anchorXC = anchor.xc;
    double anchorYC = anchor.yc;

    double boxWidth = box.x2 - box.x1;
    double boxHeight = box.y2 - box.y1;
    double boxXC = box.x1 + 0.5 * boxWidth;
    double boxYC = box.y1 + 0.5 * boxHeight;

    /* The weight factors (10.0, 5.0) are taken from detectron. */
    target.dxc = 10.0 * (boxXC - anchorXC) / anchorWidth;
    target.dyc = 10.0 * (boxYC - anchorYC) / anchorHeight;
    target.dw = 5.0 * std::log(boxWidth / anchorWidth);
    target.dh = 5.0 * std::log(boxHeight / anchorHeight);

    target.deltaValid = box.deltaValid ? 1 : 0;
    if (box.deltaValid) {
        target.deltaPrevXc = box.dxc;
        target.deltaPrevYc = box.dyc;
        target.deltaPrevW = box.dw;
        target.deltaPrevH = box.dh;
    }
}

void BBUtils::detectionToBox(const TargetBoxDetection &targetDetection,
                             const AnchorBox &anchor, BoundingBoxDetection &boxDetection) const
{
    boxDetection.score = targetDetection.objectnessScore;
    boxDetection.embedding = targetDetection.embedding;
    auto &box = boxDetection.box;
    box.cls = targetDetection.cls;
    box.depth = targetDetection.depth + anchor.depth;

    double anchorWidth = anchor.x2 - anchor.x1;
    double anchorHeight = anchor.y2 - anchor.y1;
    double anchorXC = anchor.xc;
    double anchorYC = anchor.yc;

    /* The weight factors (10.0, 5.0) are taken from detectron. */
    double boxWidth = std::exp(targetDetection.dw / 5.0) * anchorWidth;
    double boxHeight = std::exp(targetDetection.dh / 5.0) * anchorHeight;

    double boxXC = targetDetection.dxc / 10.0 * anchorWidth + anchorXC;
    double boxYC = targetDetection.dyc / 10.0 * anchorHeight + anchorYC;

    box.x1 = boxXC - 0.5 * boxWidth;
    box.y1 = boxYC - 0.5 * boxHeight;
    box.x2 = box.x1 + boxWidth;
    box.y2 = box.y1 + boxHeight;

    box.dxc = targetDetection.deltaPrevXc;
    box.dyc = targetDetection.deltaPrevYc;
    box.dw = std::exp(targetDetection.deltaPrevW);
    box.dh = std::exp(targetDetection.deltaPrevH);
    box.deltaValid = true;
}
