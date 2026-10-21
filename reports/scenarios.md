# Identity continuity evaluation

| Scenario | Confirmed* | Temporary* | Unknown* | Conf. precision | Wrong-person frames | ID switches | Recovery (s) | Centring active |
|---|---|---|---|---|---|---|---|---|
| head_turn | 77% | 23% | 0% | 1.00 | 0 | 0 | 0.00 | 88% |
| occlusion | 93% | 7% | 0% | 1.00 | 0 | 0 | 0.00 | 88% |
| path_crossing | 96% | 3% | 0% | 1.00 | 0 | 0 | - | 87% |
| long_absence | 44% | 22% | 34% | 1.00 | 0 | 0 | 0.00 | 60% |
| combined | 67% | 32% | 0% | 1.00 | 0 | 0 | 0.00 | 91% |

## head_turn

Duration 9.0s, 270 frames, faces visible in 203 frames.

| Window | State | Mean confidence |
|---|---|---|
| 0.00-1.03s | unknown | 0.00 |
| 1.03-3.50s | confirmed | 0.92 |
| 3.50-5.23s | temporary | 0.48 |
| 5.23-8.90s | confirmed | 0.93 |
| 8.90-8.97s | temporary | 0.69 |

- **PASS** confirmed observed: 184 confirmed frames
- **PASS** temporary during face loss: 55 temporary frames
- **PASS** recovered after face loss: recovery_mean=0.0s over 1 episodes
- **PASS** confirmed precision: 100.00% of confirmations correct
- **PASS** no wrong-person claims: 0 frames claimed a different person
- **PASS** identity continuity: 0 identity switches
- **PASS** single identity: 1 identity id(s) claimed for the target (1 people tracked in total)
- **PASS** centring active: active in 88% of frames

## occlusion

Duration 9.0s, 270 frames, faces visible in 239 frames.

| Window | State | Mean confidence |
|---|---|---|
| 0.00-1.03s | unknown | 0.00 |
| 1.03-4.70s | confirmed | 0.94 |
| 4.70-5.23s | temporary | 0.64 |
| 5.23-8.97s | confirmed | 0.95 |

- **PASS** confirmed observed: 223 confirmed frames
- **PASS** temporary during face loss: 16 temporary frames
- **PASS** recovered after face loss: recovery_mean=0.0s over 1 episodes
- **PASS** confirmed precision: 100.00% of confirmations correct
- **PASS** no wrong-person claims: 0 frames claimed a different person
- **PASS** identity continuity: 0 identity switches
- **PASS** single identity: 1 identity id(s) claimed for the target (2 people tracked in total)
- **PASS** centring active: active in 88% of frames

## path_crossing

Duration 8.0s, 240 frames, faces visible in 240 frames.

| Window | State | Mean confidence |
|---|---|---|
| 0.00-1.03s | unknown | 0.00 |
| 1.03-3.17s | confirmed | 0.93 |
| 3.17-3.40s | temporary | 0.70 |
| 3.40-7.97s | confirmed | 0.93 |

- **PASS** confirmed observed: 202 confirmed frames
- **PASS** temporary during face loss: 7 temporary frames
- **PASS** recovered after face loss: recovery_mean=Nones over 0 episodes
- **PASS** confirmed precision: 100.00% of confirmations correct
- **PASS** no wrong-person claims: 0 frames claimed a different person
- **PASS** identity continuity: 0 identity switches
- **PASS** single identity: 1 identity id(s) claimed for the target (2 people tracked in total)
- **PASS** centring active: active in 87% of frames

## long_absence

Duration 11.0s, 330 frames, faces visible in 149 frames.

| Window | State | Mean confidence |
|---|---|---|
| 0.00-1.03s | unknown | 0.00 |
| 1.03-2.47s | confirmed | 0.92 |
| 2.47-4.67s | temporary | 0.46 |
| 4.67-8.03s | unknown | 0.12 |
| 8.03-10.97s | confirmed | 0.93 |

- **PASS** confirmed observed: 132 confirmed frames
- **PASS** temporary during face loss: 66 temporary frames
- **PASS** recovered after face loss: recovery_mean=0.0s over 1 episodes
- **PASS** unknown reached: 132 unknown frames
- **PASS** confirmed precision: 100.00% of confirmations correct
- **PASS** no wrong-person claims: 0 frames claimed a different person
- **PASS** identity continuity: 0 identity switches
- **PASS** single identity: 1 identity id(s) claimed for the target (1 people tracked in total)
- **PASS** centring active: active in 60% of frames

## combined

Duration 11.0s, 330 frames, faces visible in 203 frames.

| Window | State | Mean confidence |
|---|---|---|
| 0.00-1.03s | unknown | 0.00 |
| 1.03-2.70s | confirmed | 0.92 |
| 2.70-3.63s | temporary | 0.59 |
| 3.63-4.43s | confirmed | 0.90 |
| 4.43-4.90s | temporary | 0.68 |
| 4.90-6.00s | confirmed | 0.91 |
| 6.00-7.03s | temporary | 0.58 |
| 7.03-10.20s | confirmed | 0.91 |
| 10.20-10.97s | temporary | 0.57 |

- **PASS** confirmed observed: 202 confirmed frames
- **PASS** temporary during face loss: 97 temporary frames
- **PASS** recovered after face loss: recovery_mean=0.0s over 3 episodes
- **PASS** confirmed precision: 100.00% of confirmations correct
- **PASS** no wrong-person claims: 0 frames claimed a different person
- **PASS** identity continuity: 0 identity switches
- **PASS** single identity: 1 identity id(s) claimed for the target (2 people tracked in total)
- **PASS** centring active: active in 91% of frames
