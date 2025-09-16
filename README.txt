mcPHASES Dataset and Variable Definitions

NOTE: The following descriptions are based on the available documentation provided with data exports, or are approximations when such documentation was incomplete or unclear. For the most accurate and up-to-date information, please consult the official support resources for each respective device.

About mcPHASES
This dataset aggregates data from multiple personal health tracking devices and surveys. Each CSV file corresponds to a specific type of measurement (e.g., Fitbit activity, glucose levels, hormone concentrations, self-reported symptoms).
All tables can be joined using the participant identifier and study day labels as keys.
* id: unique participant identifier
* day_in_study: normalized day index starting from Day 1 of each participant’s study interval


For time-spanning measurements (e.g., sleep), tables may include the following columns instead of day_in_study:
* sleep_start_day_in_study
* sleep_end_day_in_study
* start_day_in_study
* end_day_in_study


All tables also include two additional columns for temporally differentiating data:
* study_interval: whether data was collected from Interval 1 (January–April 2022) or Interval 2 (July–October 2024) of the study
* is_weekend: boolean indicator of whether the recorded day was a weekend


active_minutes.csv
This file contains the duration of physical activity categorized by intensity. Fitbit defines active minutes as the time spent doing activities that are at least moderately intense. This can include walking briskly, running, biking, and other cardio activities. Columns include:


*     sedentary: minutes of sedentary exercise completed
*     lightly: minutes of light exercise completed
*     moderately: minutes of moderate exercise completed
*     very: minutes of very active exercise completed

active_zone_minutes.csv
This file contains the time spent in different heart rate zones, which Fitbit uses to assess exercise intensity. Active Zone Minutes are awarded based on time spent in the fat burn, cardio, and peak heart rate zones. Columns include:


*     timestamp: time of the day at which the entry was logged
*     heart_zone_id: fat burn, cardio or peak to represent different heart rate zones
*     total_minutes: total minutes spent in the corresponding heart rate zone


altitude.csv 
This file contains relative altitude changes as detected by Fitbit’'s altimeter. It does not include GPS-based elevation above sea level, but instead reflects vertical movement such as climbing stairs or hiking. Columns include:


*    timestamp: time of the day at which the entry was logged
*    altitude: altitude gain in meters (and not level above sea; no GPS data)


calories.csv 
This file contains the number of calories burned over time as recorded by Fitbit. It includes both resting and active calorie expenditure. Columns include:


*     timestamp: time of the day at which the entry was logged
*     calories: number of calories burned


computed_temperature.csv
This file contains temperature readings taken by the Fitbit’s skin temperature sensor, primarily during sleep. These values are used to detect trends and deviations from a personal baseline. All temperature readings are reported in degrees Celsius. Columns include:


* sleep_start_day_in_study: day index of the sleep session relative to the start of data collection.
* sleep_start_timestamp: start time of the sleep session
* sleep_end_day_in_study: day index when the sleep session ended
* sleep_end_timestamp: end time of the sleep session
* type: type of temperature data (e.g., skin)
* temperature_samples: number of temperature readings collected throughout the night.
* nightly_temperature: calculated average nightly skin temperature
* baseline_relative_sample_sum: deviation from baseline temperature, summed over the night
* baseline_relative_sample_sum_of_squares: sum of squared deviations
* baseline_relative_nightly_standard_deviation: standard deviation of nightly temperature relative to baseline
* baseline_relative_sample_standard_deviation: standard deviation of samples taken throughout the night relative to baseline


demographic_vo2_max.csv
This file contains estimated VO2 Max values, which represent the maximum rate of oxygen consumption during exercise. Fitbit estimates VO2 Max using demographic data (age, gender, weight) and resting heart rate or exercise heart rate data. Columns include:


* demographic_vo2_max: VO2 Max estimate based on demographic and heart rate data
* demographic_vo2_max_error: associated error or uncertainty with the demographic-based VO2 Max estimate
* filtered_demographic_vo2_max: VO2 Max value after applying internal Fitbit filters or smoothing
* filtered_demographic_vo2_max_error: error value corresponding to the filtered VO2 Max estimate


distance.csv
This file contains distances covered over time collected from Fitbit. The data can be derived from steps (pedometer), GPS (when available), or both. It typically reflects walking, running, and other movement-related activities. Columns include:


*     timestamp: time of the day at which the entry was logged
*     distance: distance covered in meters


estimated_oxygen_variation.csv
This file contains data used to estimate variations in blood oxygen levels (SpO2) during sleep collected by Fitbit, based on the infrared and red light signal ratios captured by the optical sensor. Large variations might indicate breathing disturbances. Columns include:


*     timestamp: time of the day at which the entry was logged
*     infrared_to_red_signal_ratio: ratio of infrared to red light absorption, which is used to infer estimated oxygen variation (relative changes in blood oxygen saturation)


exercise.csv 
This file contains information about individual logged exercise sessions collected by Fitbit. It contains both automatically detected and manually recorded workouts. Details include duration, type of activity, intensity, and physiological metrics such as heart rate and calories burned. Columns include:


* start_day_in_study: index of the day (relative to study start) when the exercise began
* start_timestamp: start time of the exercise
* last_modified_day_in_study: index of the day when the entry was last modified
* last_modified_timestamp: last modification time of the entry
* original_start_day_in_study: original day index before any edits
* original_start_timestamp: original start time before edits
* originalduration: duration in milliseconds of the exercise session before modifications
* activityname: name of the activity (e.g., sport, walk)
* activitytypeid: internal Fitbit ID representing the activity type
* activitylevel: minutes spent at a certain activity intensity (e.g., sedentary, lightly)
* averageheartrate: average heart rate in beats per minute during the session
* calories: calories burned during the session
* duration: final duration of the session in milliseconds
* activeduration: amount of time spent in active movement during the session in milliseconds
* steps: total number of steps recorded.
* logtype: type of log (manual, auto-detected, etc.)
* manualvaluesspecified: whether the log contains manually entered or edited values 
* heartratezones: breakdown of time spent in different heart rate zones.
* activezoneminutes: number of active zone minutes earned during the session
* elevationgain: elevation gained in meters during the activity
* hasgps: whether GPS data is associated with the activity
* shouldfetchdetails: internal Fitbit check for if further detail should be fetched
* hasactivezoneminutes: indicates whether active zone minutes were available for this session


glucose.csv
This file contains continuous glucose monitoring (CGM) data exported from the Dexcom device. values are typically recorded every 5 minutes. Columns include:


*   timestamp: time of the day at which the entry was logged
*   glucose_value: glucose concentration in mmol/L measured by the sensor


heart_rate.csv
This file contains continuous heart rate measurements captured throughout the day by the Fitbit. Columns include:


*     timestamp: time of the day at which the entry was logged
*     bpm: number of heartbeats per minute
*     confidence: Fitbit-generated measure of confidence in the bpm reading


heart_rate_variability_details.csv 
This file contains heart rate variability collected by Fitbit. Heart rate variability (HRV) is the variation in the beat-to-beat interval. These include 5 minute granularity recordings of your HRV during a sleep. 


* timestamp: the start of the 5 minutes interval for which the following values were computed
* rmssd: "root mean square of successive differences", the square root of the mean of the squares of the successive differences between adjacent beat-to-beat intervals
* coverage:  the number of data points in the interval, multiplied by the mean beat-to-beat of the interval in seconds and divided by the number of seconds in the interval (300 seconds)
* low_frequency: measures long term variations in heart rate 
* high_frequency: measures short term variations in heart rate 


height_and_weight.csv
This file contains participants’ self-reported height and weight during the studies. Columns include:


* height_2022: height in centimeters, recorded in 2022
* weight_2022: weight in kilograms, recorded in 2022
* height_2024: height in centimeters, recorded in 2024
* weight_2024: weight in kilograms, recorded in 2024


hormones_and_selfreport.csv
This file combines hormone data from the Mira fertility device with daily self-reported symptom survey responses. All symptom-related responses are on a Likert-type scale from 0 ("Not at all") to 5 ("Very high"), unless otherwise specified. Columns include:


* phase: menstrual cycle phase label (e.g., follicular, ovulation, luteal, etc.) if available or derived
* lh:  luteinizing hormone (mIU/mL) level measured by the Mira device
* estrogen: estrone-3-glucuronide (a metabolite of estrogen) level (ng/mL) measured by the Mira device
* pdg: pregnanediol glucuronide (a metabolite of progesterone)  (mcg/mL) measured by the Mira device
* flow_volume: self-reported menstrual flow volume on a Likert scale (Lowest = “Not at all” to highest = “Very Heavy”)
* flow_color: categorical indicator of menstrual flow or discharge color (e.g., not at all, dark brown, bright red)
* appetite: self-reported appetite level, on a scale from 0 (not at all) to 5 (very high)
* exerciselevel: perceived exercise or physical activity level (same 0–5 scale)
* headaches: severity of headaches
* cramps: severity of menstrual or abdominal cramps
* sorebreasts: degree of breast tenderness
* fatigue: level of tiredness or exhaustion
* sleepissue: difficulty with sleep quality or staying asleep
* moodswing: frequency or severity of mood fluctuations
* stress: perceived stress level
* foodcravings: intensity of food cravings
* indigestion: presence of gastrointestinal discomfort
* bloating: extent of abdominal bloating


respiratory_rate_summary.csv
The respiratory rate (or breathing rate) is the rate at which breathing occurs collected by Fitbit. This is usually measured in breaths per minute. This file contains summaries of respiratory rate during each night's sleep:


* timestamp: the wake time
* full_sleep_breathing_rate: the average respiratory rate during a full night’s sleep
* full_sleep_standard_deviation: the amount respiratory rate varies during a full night’s sleep
* full_sleep_signal_to_noise: the signal-to-noise value for the entire sleep
* deep_sleep_breathing_rate: the respiratory rate average for deep sleep periods
* deep_sleep_standard_deviation: the amount respiratory rate varies during the deep sleep periods
* deep_sleep_signal_to_noise: the signal-to-noise value for the deep sleep periods
* light_sleep_breathing_rate: the respiratory rate average for light sleep periods
* light_sleep_standard_deviation: measures the amount respiratory rate varies during the light sleep periods
* light_sleep_signal_to_noise: the signal-to-noise value for the light sleep periods
* rem_sleep_breathing_rate: the respiratory rate average for REM sleep periods
* rem_sleep_standard_deviation: measures the amount respiratory rate varies during the REM sleep periods
* rem_sleep_signal_to_noise: the signal-to-noise value for the REM sleep periods


resting_heart_rate.csv
This file contains daily resting heart rate values collected by Fitbit, representing heart rate when the participant was awake, calm, and inactive. Columns include:


*     value: estimated resting heart rate in beats per minute
*     error: margin of error or confidence range associated with the estimate


sleep.csv
This file contains sleep session logs including timestamps, durations, and quality metrics collected by Fitbit. Columns include:


* sleep_start_day_in_study: index of the day in the study when the sleep session started
* sleep_start_timestamp: start time of the sleep session
* sleep_end_day_in_study: day index when sleep ended
* sleep_end_timestamp: end time of the sleep session
* duration: total milliseconds of the sleep session
* minutestofallasleep: minutes it took to fall asleep
* minutesasleep: total minutes spent asleep
* minutesawake: total minutes spent awake
* minutesafterwakeup: minutes spent awake after final wake time before ending the sleep session.
* timeinbed: total minutes in bed
* efficiency: percentage of time in bed spent asleep
* type: sleep log type (e.g., classic or stages)
* infocode: internal use field indicating logging or processing metadata
* levels: sleep stage details (e.g., restless, deep)
* mainsleep: boolean indicating whether this is the main sleep session of the day


sleep_score.csv
This file contains daily sleep score breakdowns provided by Fitbit, summarizing overall sleep quality based on multiple contributing factors. Columns include:


* timestamp: the time the score applies to
* overall_score: overall sleep score out of 100
* composition_score: score for sleep architecture (balance of light, deep, and REM sleep)
* revitalization_score: score for restoration, based on factors like heart rate and restlessness
* duration_score: score based on total sleep duration
* deep_sleep_in_minutes: minutes spent in deep sleep
* resting_heart_rate: resting heart rate during sleep
* restlessness: movement-based measure of sleep restlessness


steps.csv
This file contains the number of tracked steps taken throughout the day collected by Fitbit. Columns include:


*     timestamp: time of the day at which the entry was logged
*     steps: number of recorded steps


stress_score.csv
Users who have access to the Stress Management experience receive a daily Stress Management Score, which is a round number comprising 3 subscores. 


* timestamp: time of the day at which Stress Score was recorded
* stress_score: value of Stress Score
* sleep_points: points from sleep components
* max_sleep_points: maximal possible points from sleep components a user can get
* responsiveness_points: points from responsiveness components
* max_responsiveness_points: maximal points from responsiveness components a user can get
* exertion_points: points from exertion components
* max_exertion_points: maximal points from exertion components a user can get
* status: IN_PROGRESS (calculations is in progress) / READY (Stress Score was calculated)
* calculation_failed: whether stress score calculation has failed for a certain date


subject-info.csv
This file contains demographic and background survey responses collected at the start of the study. Columns include:


* birth_year: year of birth
* gender: self-identified gender
* ethnicity: self-identified race/ethnicity
* education: highest level of education attained
* employment: current employment status
* income: household income bracket
* sexually_active: whether the participant is currently sexually active (yes/no)
* self_report_menstrual_health_literacy: participant's self-assessed understanding of menstrual health (Lowest = “Non-existent, highest = “Expert”)
* age_of_first_menarche: age when participant experienced their first menstrual period


time_in_heart_rate_zones.csv
This file contains time spent in different default heart rate zones as defined by Fitbit. These zones are based on a percentage of your maximum heart rate. Columns include:


* in_default_zone_3: time spent in peak zone
* in_default_zone_2: time in cardio zone
* in_default_zone_1: time in fat burn zone
* below_default_zone_1: time below fat burn zone


wrist_temperature.csv
This file contains skin temperature variation data recorded by Fitbit, typically measured during sleep. Columns include:


*     timestamp: time of the day at which the entry was logged
*     temperature_diff_from_baseline: difference in skin temperature from personal baseline, in degrees Celsius