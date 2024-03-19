# TuSimple Lane Detection Challenge - Testing Dataset

![](assets/examples/lane_example.jpg)

## Description
The lane marking is the main component on the highway. It instructs the vehicles interactively and safely drive on the highway. Lane detection is a critical task in autonomous driving, which provides localization information to the control of the car. We provide video clips for this task, and the last frame of each clip contains labelled lanes. The video clip can help algorithms to infer better lane detection results.

## Dataset Size
2782 video clips.
Information of each clip: 20 frames for each one.

## Directory Structure:
    dataset
      |
      |----clips/
      |------|
      |------|----some_clip/
      |------|----...
      |
      |----test_tasks.json      # Test Submission Template

## Evaluation:
For each prediction of a clip, please organize the result as the same format of label data.
Also, you need to output the `lanes` according to the `h_samples` in the `test_tasks.json` for evaluation. It means we are going to evaluate points on specific image heights.

__Format__

```
{
  'raw_file': str. Clip file path
  'lanes': list. A list of lanes. For each list of one lane, there is only width index on the image.
  'run_time': list of float. The running time for each frame in the clip. The unit is millisecond.
}
```
Remember we expect at most 4 lane markings in `lanes` (current lane and left/right lanes). Feel free to output either a extra left or right lane marking when changing lane. 
